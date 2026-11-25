package com.xiaoxiaobai.xxb

import android.content.ClipData
import android.content.ClipboardManager
import android.os.Build
import android.util.Base64
import android.view.HapticFeedbackConstants
import android.webkit.JavascriptInterface
import android.widget.Toast
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/**
 * window.XxbNative for the web pages (webui/js/native.js and the bundled setup pages).
 * Called on the WebView's JavaBridge thread; answers to async calls come back through window.__xxb.
 */
class Bridge(private val activity: MainActivity, private val app: XxbApp) {
    /** Only our pages (robot console, bundled assets) may use the bridge. */
    @Volatile var allowed = true

    private val ble get() = app.ble
    private val store get() = app.store

    @JavascriptInterface
    fun status(): String = if (allowed) ble.status() else "{}"

    @JavascriptInterface
    fun send(id: Int, json: String): String = if (allowed) ble.send(id, json) else "不允许"

    @JavascriptInterface
    fun scan(on: Boolean) {
        if (!allowed) return
        if (!on) ble.scan(false) else activity.withBluetooth { ble.scan(true) }
    }

    @JavascriptInterface
    fun connect(address: String) {
        if (allowed) activity.withBluetooth { ble.connect(address) }
    }

    @JavascriptInterface
    fun disconnect() {
        if (allowed) ble.disconnect()
    }

    /** The robots this phone knows, newest first, and which one is current. */
    @JavascriptInterface
    fun robots(): String {
        if (!allowed) return "{}"
        val list = JSONArray()
        store.all().forEach { list.put(it.toJson()) }
        return JSONObject().put("current", store.currentId ?: JSONObject.NULL).put("robots", list).toString()
    }

    /** Make a saved robot current and open its Bluetooth link. */
    @JavascriptInterface
    fun select(id: String) {
        if (!allowed) return
        val robot = store.get(id) ?: return
        store.currentId = id
        robot.address?.let { address -> activity.withBluetooth { ble.connect(address) } }
    }

    /** Remember a robot found over Wi-Fi ({id, name, ip, port}); keeps its Bluetooth pairing. */
    @JavascriptInterface
    fun saveRobot(json: String): String {
        if (!allowed) return ""
        return try {
            val o = JSONObject(json)
            val id = o.getString("id")
            val robot = store.get(id) ?: Robot(id, o.optString("name", "小小白"))
            robot.name = o.optString("name").ifBlank { robot.name }
            robot.ip = o.optString("ip").ifBlank { robot.ip }
            robot.port = o.optInt("port", robot.port)
            robot.lastSeen = System.currentTimeMillis()
            store.save(robot)
            store.currentId = id
            id
        } catch (e: JSONException) {
            ""
        }
    }

    @JavascriptInterface
    fun forget(id: String) {
        if (!allowed) return
        val robot = store.get(id)
        if (robot?.address != null && robot.address.equals(ble.connectedTo(), ignoreCase = true)) ble.disconnect()
        store.remove(id)
    }

    @JavascriptInterface
    fun openConsole(url: String) {
        if (allowed) activity.openConsole(url)
    }

    @JavascriptInterface
    fun openSetup(route: String) {
        if (allowed) activity.openSetup(route)
    }

    @JavascriptInterface
    fun setTheme(theme: String) {
        store.theme = if (theme == "light") "light" else "dark"
        activity.runOnUiThread { activity.applyTheme(store.theme) }
    }

    @JavascriptInterface
    fun theme(): String = store.theme

    @JavascriptInterface
    fun haptic(kind: String) {
        activity.runOnUiThread {
            val feedback = when (kind) {
                "light" -> HapticFeedbackConstants.CLOCK_TICK
                "ok" -> if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.VIRTUAL_KEY
                else -> HapticFeedbackConstants.LONG_PRESS
            }
            activity.web.performHapticFeedback(feedback)
        }
    }

    @JavascriptInterface
    fun saveFile(name: String, mime: String, base64: String) {
        if (!allowed) return
        thread(name = "xxb-save") {
            val text = try {
                "已保存到 " + Downloads.save(activity, name, mime, Base64.decode(base64, Base64.DEFAULT))
            } catch (e: Exception) {
                "保存失败：${e.message}"
            }
            activity.runOnUiThread { Toast.makeText(activity, text, Toast.LENGTH_LONG).show() }
        }
    }

    @JavascriptInterface
    fun copy(text: String) {
        activity.runOnUiThread {
            activity.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("小小白", text))
        }
    }

    /**
     * GET a robot URL from native code (the bundled pages are https and may not fetch http).
     * Answers window.__xxb.resolve(id, {ok, status, ms, body}).
     */
    @JavascriptInterface
    fun httpCheck(id: Int, url: String, timeoutMs: Int) {
        if (!allowed) return
        thread(name = "xxb-http") {
            val started = System.currentTimeMillis()
            val out = JSONObject()
            try {
                val conn = URL(url).openConnection() as HttpURLConnection
                conn.connectTimeout = timeoutMs.coerceIn(500, 15000)
                conn.readTimeout = timeoutMs.coerceIn(500, 15000)
                conn.useCaches = false
                val code = conn.responseCode
                val body = (if (code < 400) conn.inputStream else conn.errorStream)?.use { s -> s.readBytes().let { if (it.size > 65536) it.copyOf(65536) else it } }
                conn.disconnect()
                out.put("ok", code == 200).put("status", code)
                body?.let { b -> runCatching { JSONObject(String(b, Charsets.UTF_8)) }.getOrNull()?.let { out.put("body", it) } }
            } catch (e: Exception) {
                out.put("ok", false).put("error", e.message ?: e.javaClass.simpleName)
            }
            out.put("ms", System.currentTimeMillis() - started)
            activity.callJs("resolve", id, out.toString())
        }
    }

    /** {granted, bluetooth: on|off|none} */
    @JavascriptInterface
    fun permissions(): String = JSONObject()
        .put("granted", ble.hasPermissions())
        .put("bluetooth", if (!activity.hasBluetooth()) "none" else if (ble.bluetoothOn()) "on" else "off")
        .toString()

    /** Ask for the Bluetooth permissions and to turn Bluetooth on, as needed. */
    @JavascriptInterface
    fun enableBluetooth() {
        if (allowed) activity.withBluetooth { ble.refresh() }
    }

    // ------------------------------------------------------------------ follow mode

    /** Start (or retune) following with the current robot: {fast, slow} in 0..1, cling: 跟屁虫 (unchanged when left out). */
    @JavascriptInterface
    fun followStart(json: String): String {
        if (!allowed) return "不允许"
        val robot = store.get(store.currentId) ?: store.all().firstOrNull()
        val url = robot?.url ?: return "还不知道机器人的 Wi-Fi 地址：先在机器人页面连上 Wi-Fi"
        val o = runCatching { JSONObject(json) }.getOrDefault(JSONObject())
        val fast = o.optDouble("fast", store.followFast.toDouble()).toFloat().coerceIn(0.1f, 0.8f)
        val slow = o.optDouble("slow", store.followSlow.toDouble()).toFloat().coerceIn(0.1f, fast)
        store.followFast = fast
        store.followSlow = slow
        activity.startFollow(url, fast, slow, if (o.has("cling")) o.optBoolean("cling") else null)
        return ""
    }

    @JavascriptInterface
    fun followStop() {
        if (allowed) FollowService.stop(activity)
    }

    @JavascriptInterface
    fun followStatus(): String = if (allowed) FollowService.status else "{}"

    @JavascriptInterface
    fun followFrame(): String = if (allowed) FollowService.preview ?: "{}" else "{}"

    @JavascriptInterface
    fun followFrameSeq(): Int = if (allowed) FollowService.previewSeq else 0

    /** {fast, slow, battery_exempt} */
    @JavascriptInterface
    fun followSettings(): String = JSONObject()
        .put("fast", store.followFast.toDouble())
        .put("slow", store.followSlow.toDouble())
        .put("battery_exempt", activity.batteryExempt())
        .toString()

    /** Ask Android not to put the app to sleep (follow mode with the screen off). */
    @JavascriptInterface
    fun askBatteryExempt() {
        if (allowed) activity.runOnUiThread { activity.askBatteryExempt() }
    }

    /** The app's system settings page (vendor "auto start" / background switches live near it). */
    @JavascriptInterface
    fun openAppSettings() {
        if (allowed) activity.runOnUiThread { activity.openAppSettings() }
    }

    @JavascriptInterface
    fun appInfo(): String = JSONObject()
        .put("version", BuildConfig.VERSION_NAME)
        .put("android", Build.VERSION.RELEASE)
        .put("model", Build.MODEL)
        .toString()
}
