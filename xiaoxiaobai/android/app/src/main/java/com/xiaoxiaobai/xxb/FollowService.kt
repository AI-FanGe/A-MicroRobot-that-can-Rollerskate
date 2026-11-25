package com.xiaoxiaobai.xxb

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlin.math.roundToInt

/**
 * Follow mode, as a foreground service so it keeps running with the screen off: reads the
 * ESP32's camera straight over Wi-Fi, finds the user's legs with [PoseDetector] and drives the
 * robot through its HTTP API (/mc/locomotion/velocity, source "follow") at 8 Hz. The robot
 * stops by itself within [SOURCE_TIMEOUT_MS] when the commands stop coming.
 */
class FollowService : Service() {
    @Volatile private var running = false
    @Volatile private var reader: MjpegReader? = null
    private var api: RobotApi? = null
    private val controller = FollowController()
    private val cover = CoverWatch()  // guarded by controller
    @Volatile private var awaySeenAt = -1L  // the person's whole or lower legs last in view (frame time)
    @Volatile private var hugging = false
    @Volatile private var luma = 0f
    private val frames = LinkedBlockingQueue<Frame>()  // only the newest camera frame
    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null
    private val main = Handler(Looper.getMainLooper())

    // Shown on the page and in the notification.
    @Volatile private var phase = "starting"
    @Volatile private var message = "正在连接机器人…"
    @Volatile private var mode: String? = null
    @Volatile private var cameraFps = 0f
    @Volatile private var visionFps = 0f
    @Volatile private var inferMs = 0
    @Volatile private var people = 0
    @Volatile private var model = "s"
    private var notified = ""
    private var notifiedAt = 0L

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            if (running) synchronized(controller) { controller.stopping = true } else stopSelf()
            return START_NOT_STICKY
        }
        val base = intent?.getStringExtra(EXTRA_ROBOT)
        if (base == null) {
            stopSelf()
            return START_NOT_STICKY
        }
        synchronized(controller) {
            controller.fast = intent.getFloatExtra(EXTRA_FAST, controller.fast)
            controller.slow = intent.getFloatExtra(EXTRA_SLOW, controller.slow)
            if (intent.hasExtra(EXTRA_CLING)) controller.cling = intent.getBooleanExtra(EXTRA_CLING, false)
            else if (!running) controller.cling = false
            controller.stopping = false
        }
        startForegroundNow()
        if (!running) {
            running = true
            synchronized(controller) {
                cover.reset()
                controller.hugging = false
            }
            hugging = false
            awaySeenAt = -1L
            api = RobotApi(base)
            acquireLocks()
            thread(name = "follow-camera") { cameraLoop() }
            thread(name = "follow-vision") { visionLoop() }
            thread(name = "follow-control") { controlLoop() }
        }
        publish()
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        val wasMoving = running && !controller.still()
        val wasHugging = running && hugging
        running = false
        reader?.close()
        if (wasMoving) api?.let { a -> thread { runCatching { a.velocity(0f, 0f, 0f) } } }
        if (wasHugging) api?.let { a -> thread { runCatching { a.hug(false) } } }
        wakeLock?.takeIf { it.isHeld }?.release()
        wifiLock?.takeIf { it.isHeld }?.release()
        status = JSONObject().put("running", false).toString()
        preview = null
        previewSeq++
        super.onDestroy()
    }

    // ------------------------------------------------------------------ camera

    private fun cameraLoop() {
        var ip: String? = null
        while (running) {
            try {
                if (ip == null) {
                    ip = api!!.esp32Ip()
                    if (ip == null) {
                        setPhase("no_camera", "机器人还不知道摄像头（ESP32）的 Wi-Fi 地址：确认 ESP32 连上了同一个热点")
                        sleep(2000)
                        continue
                    }
                }
                MjpegReader("http://$ip:81/stream?size=qvga&quality=12").use { r ->
                    reader = r
                    var count = 0
                    var since = SystemClock.elapsedRealtime()
                    while (running) {
                        val jpeg = r.next()
                        val now = SystemClock.elapsedRealtime()
                        frames.clear()
                        frames.offer(Frame(jpeg, now - CAMERA_LAG_MS))
                        count++
                        if (now - since >= 2000) {
                            cameraFps = count * 1000f / (now - since)
                            count = 0
                            since = now
                        }
                    }
                }
            } catch (e: Exception) {
                if (!running) break
                cameraFps = 0f
                val where = ip?.let { "（$it）" }.orEmpty()
                setPhase("no_camera", "摄像头连不上$where：${e.message ?: e.javaClass.simpleName}")
                ip = null
                sleep(1000)
            } finally {
                reader = null
            }
        }
    }

    // ------------------------------------------------------------------ vision

    private fun loadDetector(name: String): PoseDetector? = try {
        PoseDetector(this, name).also { model = if (name == PoseDetector.MODEL_FAST) "n" else "s" }
    } catch (e: Exception) {
        setPhase("error", "识别模型加载失败：${e.message}")
        null
    }

    private fun visionLoop() {
        var detector = loadDetector(PoseDetector.MODEL) ?: loadDetector(PoseDetector.MODEL_FAST) ?: return
        try {
            var count = 0
            var since = SystemClock.elapsedRealtime()
            var lastRun = 0L
            var avgMs = 0f
            var runs = 0
            while (running) {
                sleep(MIN_FRAME_MS - (SystemClock.elapsedRealtime() - lastRun))
                val frame = frames.poll(500, TimeUnit.MILLISECONDS) ?: continue
                lastRun = SystemClock.elapsedRealtime()
                val bmp = BitmapFactory.decodeByteArray(frame.jpeg, 0, frame.jpeg.size) ?: continue
                val (mean, std) = brightness(bmp)
                luma = std
                val found = try {
                    detector.detect(bmp)
                } catch (e: Exception) {
                    setPhase("error", "识别出错：${e.message}")
                    emptyList()
                } finally {
                    bmp.recycle()
                }
                val now = SystemClock.elapsedRealtime()
                inferMs = (now - lastRun).toInt()
                people = found.size
                val target = synchronized(controller) {
                    cover.frame(mean, std, frame.atMs)
                    val expected = controller.expectedX(frame.atMs)
                    pickTarget(found, expected).also { t ->
                        controller.see(t?.let { Sighting((it.legX(expected ?: 0.5f) - 0.5f) * 2f, it.legs, frame.atMs) })
                    }
                }
                if (target != null && (target.legs == Legs.FULL || target.legs == Legs.LOWER)) awaySeenAt = frame.atMs
                preview = previewJson(frame.jpeg, found, target)
                previewSeq++
                count++
                if (now - since >= 2000) {
                    visionFps = count * 1000f / (now - since)
                    count = 0
                    since = now
                }
                // The accurate model when the phone keeps up with it, else the light one.
                avgMs = if (runs == 0) inferMs.toFloat() else avgMs * 0.9f + inferMs * 0.1f
                if (++runs >= 20 && avgMs > SLOW_INFER_MS && detector.model != PoseDetector.MODEL_FAST) {
                    val fast = loadDetector(PoseDetector.MODEL_FAST)
                    if (fast != null) {
                        detector.close()
                        detector = fast
                        runs = 0
                    }
                }
            }
        } finally {
            detector.close()
        }
    }

    /** Brightness mean and spread of the frame, from a 32x24 thumbnail. */
    private fun brightness(bmp: Bitmap): Pair<Float, Float> {
        val small = Bitmap.createScaledBitmap(bmp, THUMB_W, THUMB_H, true)
        val px = IntArray(THUMB_W * THUMB_H)
        small.getPixels(px, 0, THUMB_W, 0, 0, THUMB_W, THUMB_H)
        if (small !== bmp) small.recycle()
        return lumaStats(px)
    }

    private fun previewJson(jpeg: ByteArray, found: List<Person>, target: Person?): String {
        fun box(p: Person) = JSONArray(listOf(p.x1, p.y1, p.x2, p.y2).map { it.toDouble() })
        val o = JSONObject()
            .put("jpeg", android.util.Base64.encodeToString(jpeg, android.util.Base64.NO_WRAP))
            .put("others", JSONArray(found.filter { it !== target }.map { box(it) }))
        if (target != null) {
            o.put("target", JSONObject()
                .put("box", box(target))
                .put("kpts", JSONArray(target.kpts.map { it.toDouble() }))
                .put("legs", target.legs.name))
        }
        return o.toString()
    }

    // ------------------------------------------------------------------ control

    private fun controlLoop() {
        val api = api!!
        var registered = false
        var walking = false
        var lastModeCheck = 0L
        var walkAsked = 0L
        var last = SystemClock.elapsedRealtime()
        var stopSince = 0L
        while (running) {
            val now = SystemClock.elapsedRealtime()
            val dt = ((now - last) / 1000f).coerceIn(0f, 0.5f)
            last = now
            val stopping = synchronized(controller) { controller.stopping }
            if (stopping && stopSince == 0L) stopSince = now
            if (stopping && (!walking || now - stopSince > STOP_TIMEOUT_MS)) break
            try {
                if (!registered) {
                    api.registerSource()
                    registered = true
                }
                if (!walking) {
                    if (now - lastModeCheck > 1500) {
                        lastModeCheck = now
                        val current = api.action()
                        mode = current.optString("label")
                        when (current.optString("name")) {
                            in WALKING -> {
                                walking = true
                                synchronized(controller) { controller.reset() }
                            }
                            STAND -> {
                                if (now - walkAsked > 5000) {
                                    walkAsked = now
                                    api.setAction(WALK)
                                }
                                setPhase("waiting", "机器人站着，正在切到「走跑」…")
                            }
                            else -> setPhase("waiting", "机器人现在是「${mode}」：先在控制台让它站立，站好后会自动开始跟随")
                        }
                    }
                    if (!walking) {
                        sleep(CONTROL_MS)
                        continue
                    }
                }
                hugStep(api, now, stopping)
                val cmd = synchronized(controller) { controller.tick(now, dt) }
                val res = api.velocity(cmd.vx, cmd.vtheta, cmd.headYaw)
                if (res.optJSONObject("header")?.optInt("code", 1) == 0) {
                    // An older robot, or the head busy with something else: it stays ahead.
                    val head = res.optJSONObject("data")?.let { !it.isNull("head_yaw") && it.has("head_yaw") } ?: false
                    synchronized(controller) { controller.headSupported = head }
                    describe(cmd, stopping)
                } else {
                    val reason = res.optJSONObject("state")?.optString("reason_name").orEmpty()
                    val text = res.optString("message").ifEmpty { "机器人拒绝了速度指令" }
                    synchronized(controller) { controller.reset() }
                    when (reason) {
                        "REASON_ACTION" -> {
                            walking = false
                            setPhase("waiting", text)
                        }
                        "REASON_NOT_FOUND" -> registered = false
                        "REASON_INPUT_SOURCE_BUSY" -> setPhase("busy", "$text，跟随暂停")
                        else -> setPhase("error", text)
                    }
                }
                if (stopping && synchronized(controller) { controller.still() }) {
                    api.velocity(0f, 0f, 0f)
                    break
                }
            } catch (e: IOException) {
                synchronized(controller) { controller.reset() }
                registered = false
                walking = false
                setPhase("offline", "连不上机器人：${e.message ?: "网络错误"}")
                sleep(1000)
                continue
            }
            sleep(CONTROL_MS - (SystemClock.elapsedRealtime() - now))
        }
        if (hugging) runCatching { letGo(api, SystemClock.elapsedRealtime()) }
        running = false
        reader?.close()
        main.post { stopSelf() }
    }

    private var hugEndedAt = Long.MIN_VALUE

    /**
     * 跟屁虫: hug right after their legs were made out (a fresh sighting since the last hug) once the
     * camera is covered or the robot has crept on towards the lost leg for [FollowController.CREEP_HUG_MS];
     * let go when they walk off, the camera clear again ([hugOver]), or on stop.
     */
    private fun hugStep(api: RobotApi, now: Long, stopping: Boolean) {
        val cling: Boolean
        val covered: Boolean
        val clear: Long
        val lastCovered: Long
        val close: Boolean
        val crept: Long
        synchronized(controller) {
            cling = controller.cling
            covered = cover.covered(now)
            clear = cover.clearFor(now)
            lastCovered = cover.lastCovered
            close = controller.legsRecently(now, FollowController.APPROACH_MS, hugEndedAt)
            crept = controller.creptMs(now)
        }
        if (hugging) {
            if (stopping || !cling || hugOver(clear, awaySeenAt > lastCovered)) letGo(api, now)
            return
        }
        if (!cling || stopping || !close) return
        if (!covered && crept < FollowController.CREEP_HUG_MS) return
        synchronized(controller) {
            controller.reset()
            controller.hugging = true
        }
        hugging = true
        api.velocity(0f, 0f, 0f)
        setPhase("hugging", "贴上啦，抱住你！")
        val res = try {
            api.hug(true)
        } catch (e: IOException) {
            endHug(now)
            throw e
        }
        if (res.optJSONObject("header")?.optInt("code", 1) != 0) {
            endHug(now)
            setPhase("error", res.optString("message").ifEmpty { "机器人没能抱住" })
        }
    }

    private fun letGo(api: RobotApi, now: Long) {
        endHug(now)
        api.hug(false)
    }

    private fun endHug(now: Long) {
        hugging = false
        hugEndedAt = now
        synchronized(controller) { controller.hugging = false }
    }

    private fun describe(cmd: FollowCommand, stopping: Boolean) {
        val mode: FollowMode
        val legs: Legs?
        val side: String
        val head: Boolean
        val crept: Long
        synchronized(controller) {
            mode = controller.mode
            legs = controller.legs
            side = if (controller.searchDir > 0f) "左" else "右"
            head = controller.headSupported
            crept = controller.creptMs(SystemClock.elapsedRealtime())
        }
        val cling = synchronized(controller) { controller.cling }
        val (p, text) = when {
            stopping -> "stopping" to "正在慢慢停下…"
            mode == FollowMode.HUG -> "hugging" to "抱着你的腿不放～走开它就松手继续跟"
            mode == FollowMode.APPROACH -> "close" to
                "贴着腿往前蹭… ${"%.1f".format((FollowController.CREEP_HUG_MS - crept).coerceAtLeast(0L) / 1000f)} 秒后抱住"
            mode == FollowMode.LOOK -> "searching" to if (head) "刚看不到你，转头往${side}边找" else "刚看不到你，往${side}边转身找"
            mode == FollowMode.SEARCH -> "searching" to "正在向${side}转身找你"
            mode == FollowMode.SCAN -> "searching" to if (head) "左右转头找你" else "等你出现在机器人前面"
            legs == Legs.FULL -> "following" to if (cmd.targetVx > 0.005f) "看到整条腿，跟进中" else "看到你了，先转过去"
            legs == Legs.LOWER -> "close" to if (cmd.targetVx > 0.005f) "只看到小腿，慢速跟进" else "看到你了，先转过去"
            legs == Legs.FEET -> "close" to if (cling) "很近了，慢慢贴上去" else "太近了，停下等你"
            else -> "searching" to "看到人但看不到腿"
        }
        setPhase(p, text, cmd)
    }

    // ------------------------------------------------------------------ status

    private fun setPhase(p: String, text: String, cmd: FollowCommand? = null) {
        phase = p
        message = text
        publish(cmd)
    }

    @Synchronized
    private fun publish(cmd: FollowCommand? = null) {
        val o = JSONObject()
            .put("running", running)
            .put("phase", phase)
            .put("message", message)
            .put("mode", mode ?: JSONObject.NULL)
            .put("legs", controller.legs?.label ?: JSONObject.NULL)
            .put("vx", (controller.vx * 100).roundToInt())
            .put("vtheta", (controller.vtheta * 100).roundToInt())
            .put("target_vx", cmd?.let { (it.targetVx * 100).roundToInt() } ?: JSONObject.NULL)
            .put("fast", (controller.fast * 100).roundToInt())
            .put("slow", (controller.slow * 100).roundToInt())
            .put("camera_fps", (cameraFps * 10).roundToInt() / 10.0)
            .put("vision_fps", (visionFps * 10).roundToInt() / 10.0)
            .put("infer_ms", inferMs)
            .put("people", people)
            .put("model", model)
            .put("head", controller.headYaw.roundToInt())
            .put("head_ok", controller.headSupported)
            .put("bearing", controller.bearing?.roundToInt() ?: JSONObject.NULL)
            .put("cling", controller.cling)
            .put("hug", hugging)
            .put("luma_std", luma.roundToInt())
        status = o.toString()
        // Android drops notification updates past a few a second.
        val line = "$message · 速度 ${(controller.vx * 100).roundToInt()}%"
        val now = SystemClock.elapsedRealtime()
        if (line != notified && now - notifiedAt > 1500) {
            notified = line
            notifiedAt = now
            getSystemService(NotificationManager::class.java)?.notify(NOTIFY_ID, notification(line))
        }
    }

    private fun startForegroundNow() {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "跟随模式", NotificationManager.IMPORTANCE_LOW).apply {
                description = "跟随模式运行时常驻，熄屏也继续跟随"
            })
        }
        val type = if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE else 0
        ServiceCompat.startForeground(this, NOTIFY_ID, notification(message), type)
    }

    private fun notification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, FollowService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentTitle(if (controller.cling) "小小白 · 跟屁虫" else "小小白 · 跟随模式")
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(open)
            .addAction(0, "停止跟随", stop)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    private fun acquireLocks() {
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "xxb:follow").apply {
                setReferenceCounted(false)
                acquire()
            }
        val wifiMode = if (Build.VERSION.SDK_INT >= 29) WifiManager.WIFI_MODE_FULL_LOW_LATENCY
        else @Suppress("DEPRECATION") WifiManager.WIFI_MODE_FULL_HIGH_PERF
        wifiLock = (applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager)
            .createWifiLock(wifiMode, "xxb:follow").apply {
                setReferenceCounted(false)
                acquire()
            }
    }

    private fun sleep(ms: Long) {
        if (ms > 0) runCatching { Thread.sleep(ms) }
    }

    companion object {
        private const val ACTION_START = "com.xiaoxiaobai.xxb.FOLLOW_START"
        private const val ACTION_STOP = "com.xiaoxiaobai.xxb.FOLLOW_STOP"
        private const val EXTRA_ROBOT = "robot"
        private const val EXTRA_FAST = "fast"
        private const val EXTRA_SLOW = "slow"
        private const val EXTRA_CLING = "cling"
        private const val THUMB_W = 32
        private const val THUMB_H = 24
        private const val CHANNEL = "follow"
        private const val NOTIFY_ID = 7
        private const val CONTROL_MS = 100L
        private const val MIN_FRAME_MS = 50L  // at most ~20 detections a second
        private const val SLOW_INFER_MS = 110f  // the accurate model takes longer than this: use the light one
        private const val CAMERA_LAG_MS = 60L  // exposure, JPEG and Wi-Fi: a frame shows the past
        private const val STOP_TIMEOUT_MS = 8000L
        const val SOURCE = "follow"
        const val SOURCE_TIMEOUT_MS = 600
        private const val STAND = "BIPED_STAND_DEFAULT"
        private const val WALK = "BIPED_WALK_RUN"
        private val WALKING = setOf(WALK, "BIPED_WALK_UPPER", "BIPED_SKATE")

        /** {running, phase, message, legs, vx, ...} for the follow page. */
        @Volatile var status: String = JSONObject().put("running", false).toString()
            private set

        /** The last analysed frame and what was found in it: {jpeg (base64), target?, others}. */
        @Volatile var preview: String? = null
            private set

        /** Bumped with each new [preview], so the page only decodes new ones. */
        @Volatile var previewSeq = 0
            private set

        /** cling: 跟屁虫, which walks right up to the person and hugs their leg (null: as it was). */
        fun start(context: Context, robotUrl: String, fast: Float, slow: Float, cling: Boolean? = null) {
            val intent = Intent(context, FollowService::class.java).setAction(ACTION_START)
                .putExtra(EXTRA_ROBOT, robotUrl).putExtra(EXTRA_FAST, fast).putExtra(EXTRA_SLOW, slow)
            if (cling != null) intent.putExtra(EXTRA_CLING, cling)
            ContextCompat.startForegroundService(context, intent)
        }

        /** Ramp down to a stop, then end the service. */
        fun stop(context: Context) {
            context.startService(Intent(context, FollowService::class.java).setAction(ACTION_STOP))
        }
    }
}

/** A camera frame and when it was taken (elapsedRealtime). */
private class Frame(val jpeg: ByteArray, val atMs: Long)

/** The few robot API calls the follow mode makes (src/api/server.py), JSON in and out. */
private class RobotApi(base: String) {
    private val root = base.trimEnd('/') + "/api/v1"

    fun registerSource() {
        val body = JSONObject().put("name", FollowService.SOURCE).put("priority", 50)
            .put("timeout_ms", FollowService.SOURCE_TIMEOUT_MS).put("head_front", true)
            .put("description", "手机 App 跟随模式（识别腿部）")
        val res = call("POST", "/mc/input_source", body)
        if (res.optJSONObject("header")?.optInt("code", 1) != 0) throw IOException(res.optString("message", "注册输入源失败"))
    }

    /** {name, label} of the robot's current mode. */
    fun action(): JSONObject =
        call("GET", "/mc/action", null).optJSONObject("data")?.optJSONObject("current_action") ?: JSONObject()

    fun setAction(name: String) = call("POST", "/mc/action", JSONObject().put("action", name).put("source", "app_proxy"))

    fun velocity(vx: Float, vtheta: Float, headYaw: Float): JSONObject = call(
        "POST", "/mc/locomotion/velocity",
        JSONObject().put("source", FollowService.SOURCE).put("vx", vx.toDouble()).put("vy", 0)
            .put("vtheta", vtheta.toDouble()).put("head_yaw", headYaw.toDouble()),
    )

    /** Hug the leg in front / let go (/mc/hug): switching to 「走跑·上肢自由」 first takes a few seconds. */
    fun hug(enabled: Boolean): JSONObject =
        call("POST", "/mc/hug", JSONObject().put("enabled", enabled).put("source", FollowService.SOURCE), HUG_TIMEOUT_MS)

    fun esp32Ip(): String? =
        call("GET", "/hal/camera", null).optJSONObject("data")?.optString("esp32_ip")?.takeIf { it.isNotEmpty() && it != "null" }

    // Bodies are read to the end and the connection is not disconnected, so it can be reused.
    private fun call(method: String, path: String, body: JSONObject?, readTimeout: Int = 2500): JSONObject {
        val conn = URL(root + path).openConnection() as HttpURLConnection
        conn.requestMethod = method
        conn.connectTimeout = 2000
        conn.readTimeout = readTimeout
        conn.useCaches = false
        if (body != null) {
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json")
            conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
        }
        val code = conn.responseCode
        val text = (if (code < 400) conn.inputStream else conn.errorStream)?.use { String(it.readBytes(), Charsets.UTF_8) }.orEmpty()
        return runCatching { JSONObject(text) }.getOrElse { throw IOException("HTTP $code") }
    }

    companion object {
        const val HUG_TIMEOUT_MS = 9000
    }
}
