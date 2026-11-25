package com.xiaoxiaobai.xxb

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.BluetoothStatusCodes
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.ParcelUuid
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/**
 * The phone's side of the robot's Bluetooth link (src/api/ble.py): scan, connect, pair, and
 * carry the web pages' requests and the robot's events. Everything runs on one thread.
 */
@SuppressLint("MissingPermission")
class BleManager(private val context: Context, private val store: RobotStore) {

    interface Listener {
        fun onState(json: String)
        fun onReply(jsId: Int, json: String)
        fun onEvent(json: String)
        fun onScan(json: String)
    }

    @Volatile var listener: Listener? = null

    /** Bumped when the WebView loads a page: replies to the old page are dropped. */
    @Volatile var generation = 0
        private set

    private val handler = Handler(HandlerThread("xxb-ble").apply { start() }.looper)
    private val adapter: BluetoothAdapter? get() = context.getSystemService(BluetoothManager::class.java)?.adapter

    private enum class Phase { IDLE, CONNECTING, HELLO, LINKED }

    private class Waiter(val jsId: Int, val generation: Int, val expires: Long, val internal: ((JSONObject) -> Unit)?)

    private sealed class Op {
        class Read(val c: BluetoothGattCharacteristic, val done: (ByteArray?) -> Unit) : Op()
        class Notify(val c: BluetoothGattCharacteristic, val done: (Boolean) -> Unit) : Op()
        class Write(val data: ByteArray, val key: String?) : Op()
    }

    private var target: String? = null
    private var gatt: BluetoothGatt? = null
    private var rx: BluetoothGattCharacteristic? = null
    private var phase = Phase.IDLE
    private var discovering = false
    private var mtu = 23
    private var error: String? = null
    private var robot: JSONObject? = null
    private var authed = false
    private var setup = false
    private var retries = 0
    private var retryPending = false
    private var eventsWanted = false
    private val assembler = Assembler()
    private var seq = 0
    private var nextId = 1
    private val waiting = HashMap<Int, Waiter>()
    private val ops = ArrayDeque<Op>()
    private var current: Op? = null
    private var busyRetries = 0
    private var scanning = false
    private val found = LinkedHashMap<String, JSONObject>()
    private var scanPosted = false

    @Volatile private var statusJson = "{\"state\":\"idle\"}"
    @Volatile private var linked = false
    @Volatile private var ready = false

    init {
        ContextCompat.registerReceiver(context, object : BroadcastReceiver() {
            override fun onReceive(c: Context, intent: Intent) {
                val state = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, -1)
                handler.post {
                    when (state) {
                        BluetoothAdapter.STATE_ON -> if (target != null && gatt == null) open() else publish()
                        BluetoothAdapter.STATE_OFF -> {
                            scanning = false
                            close()
                            publish()
                        }
                    }
                }
            }
        }, IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED), ContextCompat.RECEIVER_NOT_EXPORTED)
        handler.post { publish() }
    }

    // ------------------------------------------------------------------ public, any thread

    fun hasPermissions() = PERMISSIONS.all { ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED }

    fun bluetoothOn() = adapter?.isEnabled == true

    fun status(): String = statusJson

    /** Re-check permissions and the adapter (after a permission dialog or the Bluetooth switch). */
    fun refresh() = handler.post {
        if (target != null && gatt == null && !retryPending) open() else publish()
    }

    fun connect(address: String) = handler.post {
        val wanted = address.uppercase()
        error = null
        if (wanted == target && (gatt != null || retryPending)) {
            publish()
            return@post
        }
        close()
        target = wanted
        retries = 0
        robot = store.byAddress(wanted)?.let { JSONObject().put("id", it.id).put("name", it.name).put("ip", it.ip ?: JSONObject.NULL).put("port", it.port) }
        open()
    }

    fun disconnect() = handler.post {
        target = null
        error = null
        robot = null
        close()
        publish()
    }

    fun connectedTo(): String? = target

    fun scan(on: Boolean) = handler.post {
        if (on) startScan() else stopScan()
        publish()
    }

    /** A message from a page (native.js send); the reply goes to listener.onReply(jsId). Returns an error or "". */
    fun send(jsId: Int, json: String): String {
        val msg = try {
            JSONObject(json)
        } catch (e: JSONException) {
            return "消息格式不对"
        }
        val type = msg.optString("t")
        val path = msg.optString("p").substringBefore('?')
        if (!linked) return "蓝牙未连接"
        if (type == "req" && !ready && path !in OPEN_PATHS) return "还没有和机器人配对"
        val gen = generation
        handler.post {
            if (phase != Phase.LINKED) {
                listener?.onReply(jsId, JSONObject().put("error", "蓝牙未连接").toString())
                return@post
            }
            if (type == "sub") eventsWanted = msg.optBoolean("on")
            val quiet = msg.optInt("q") == 1
            val key = if (type == "req" && path == VELOCITY) "velocity" else null
            request(msg, key, jsId = jsId, gen = gen, ttl = if (quiet) 5_000 else 120_000)
        }
        return ""
    }

    fun pageChanged() {
        generation++
        handler.post {
            waiting.entries.removeAll { it.value.internal == null }
            if (eventsWanted) {
                eventsWanted = false
                if (phase == Phase.LINKED) request(JSONObject().put("t", "sub").put("on", false), null, internal = {})
            }
        }
    }

    // ------------------------------------------------------------------ connection

    private fun open() {
        handler.removeCallbacks(retry)
        retryPending = false
        val address = target ?: return
        val a = adapter
        if (a == null || !a.isEnabled || !hasPermissions()) {
            publish()
            return
        }
        val device = try {
            a.getRemoteDevice(address)
        } catch (e: IllegalArgumentException) {
            target = null
            error = "蓝牙地址不对"
            publish()
            return
        }
        phase = Phase.CONNECTING
        discovering = false
        publish()
        gatt = device.connectGatt(context, false, callback, BluetoothDevice.TRANSPORT_LE)
        handler.postDelayed(linkTimeout, 20_000)
    }

    private val linkTimeout = Runnable { if (phase != Phase.LINKED) lost("连接机器人超时：确认机器人开着、离手机不远") }
    private val retry = Runnable { open() }
    private val mtuFallback = Runnable { gatt?.let(::discover) }

    private fun close() {
        handler.removeCallbacks(linkTimeout)
        handler.removeCallbacks(retry)
        handler.removeCallbacks(opTimeout)
        handler.removeCallbacks(mtuFallback)
        retryPending = false
        gatt?.let {
            try {
                it.disconnect()
                it.close()
            } catch (e: Exception) {
                Log.w(TAG, "close: $e")
            }
        }
        gatt = null
        rx = null
        phase = Phase.IDLE
        authed = false
        mtu = 23
        ops.clear()
        current = null
        assembler.reset()
        failWaiting("蓝牙连接断开")
    }

    private fun lost(reason: String) {
        Log.i(TAG, "link lost: $reason")
        close()
        if (target != null) {
            error = reason
            retryPending = true
            handler.postDelayed(retry, minOf(15_000L, 1000L shl minOf(retries, 4)))
            retries++
        }
        publish()
    }

    private fun discover(g: BluetoothGatt) {
        if (discovering) return
        discovering = true
        if (!g.discoverServices()) lost("读取机器人的蓝牙服务失败")
    }

    private val callback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(g: BluetoothGatt, status: Int, newState: Int) {
            handler.post {
                if (g !== gatt) {
                    g.close()
                    return@post
                }
                if (newState == BluetoothProfile.STATE_CONNECTED && status == BluetoothGatt.GATT_SUCCESS) {
                    g.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH)
                    if (g.requestMtu(517)) handler.postDelayed(mtuFallback, 3000) else discover(g)
                } else {
                    lost(if (status == 8) "和机器人的蓝牙断开了（走远了或者机器人关机了）" else "蓝牙连接断开（$status）")
                }
            }
        }

        override fun onMtuChanged(g: BluetoothGatt, value: Int, status: Int) {
            handler.post {
                if (g !== gatt) return@post
                handler.removeCallbacks(mtuFallback)
                if (status == BluetoothGatt.GATT_SUCCESS) mtu = value
                discover(g)
            }
        }

        override fun onServicesDiscovered(g: BluetoothGatt, status: Int) {
            handler.post {
                if (g !== gatt) return@post
                val service = g.getService(Protocol.SERVICE)
                val rxc = service?.getCharacteristic(Protocol.RX)
                val txc = service?.getCharacteristic(Protocol.TX)
                if (rxc == null || txc == null) {
                    target = null
                    error = "这台设备不是小小白"
                    close()
                    publish()
                    return@post
                }
                rx = rxc
                phase = Phase.HELLO
                service.getCharacteristic(Protocol.INFO)?.let { info -> enqueue(Op.Read(info) { data -> data?.let(::parseInfo) }) }
                enqueue(Op.Notify(txc) { ok -> if (ok) hello() else lost("订阅机器人消息失败") })
            }
        }

        override fun onCharacteristicRead(g: BluetoothGatt, c: BluetoothGattCharacteristic, value: ByteArray, status: Int) {
            handler.post { if (g === gatt) finish(value, status == BluetoothGatt.GATT_SUCCESS) }
        }

        @Deprecated("Android 12 and older")
        override fun onCharacteristicRead(g: BluetoothGatt, c: BluetoothGattCharacteristic, status: Int) {
            if (Build.VERSION.SDK_INT >= 33) return
            @Suppress("DEPRECATION") val value = c.value?.copyOf()
            handler.post { if (g === gatt) finish(value, status == BluetoothGatt.GATT_SUCCESS) }
        }

        override fun onCharacteristicWrite(g: BluetoothGatt, c: BluetoothGattCharacteristic, status: Int) {
            handler.post { if (g === gatt) finish(null, status == BluetoothGatt.GATT_SUCCESS) }
        }

        override fun onDescriptorWrite(g: BluetoothGatt, d: BluetoothGattDescriptor, status: Int) {
            handler.post { if (g === gatt) finish(null, status == BluetoothGatt.GATT_SUCCESS) }
        }

        override fun onCharacteristicChanged(g: BluetoothGatt, c: BluetoothGattCharacteristic, value: ByteArray) {
            val packet = value.copyOf()
            handler.post { if (g === gatt) assembler.feed(packet)?.let(::onMessage) }
        }

        @Deprecated("Android 12 and older")
        override fun onCharacteristicChanged(g: BluetoothGatt, c: BluetoothGattCharacteristic) {
            if (Build.VERSION.SDK_INT >= 33) return
            @Suppress("DEPRECATION") val packet = c.value?.copyOf() ?: return
            handler.post { if (g === gatt) assembler.feed(packet)?.let(::onMessage) }
        }
    }

    private fun parseInfo(data: ByteArray) {
        try {
            robot = JSONObject(String(data, Charsets.UTF_8))
            setup = robot?.optBoolean("setup") == true
        } catch (e: JSONException) {
            Log.w(TAG, "bad INFO: $e")
        }
    }

    private fun hello() {
        val id = robot?.optString("id")?.takeIf { it.isNotEmpty() }
        val saved = store.get(id) ?: store.byAddress(target)
        val msg = JSONObject().put("t", "hello").put("client", store.clientId).put("name", deviceName()).put("mtu", mtu)
        saved?.token?.let { msg.put("token", it) }
        request(msg, null, internal = ::onHello)
    }

    private fun onHello(reply: JSONObject) {
        handler.removeCallbacks(linkTimeout)
        authed = reply.optBoolean("auth")
        setup = reply.optBoolean("setup")
        reply.optJSONObject("robot")?.let { robot = it }
        phase = Phase.LINKED
        retries = 0
        error = null
        remember(reply.optString("token"))
        if (authed && eventsWanted) request(JSONObject().put("t", "sub").put("on", true), null, internal = {})
        publish()
    }

    /** Save what we learnt about this robot: Bluetooth address, Wi-Fi address, a new token. */
    private fun remember(token: String?) {
        val info = robot ?: return
        val id = info.optString("id").takeIf { it.isNotEmpty() } ?: return
        val r = store.get(id) ?: Robot(id, info.optString("name", "小小白"))
        r.name = info.optString("name").takeIf { it.isNotEmpty() } ?: r.name
        r.address = target ?: r.address
        info.optString("ip").takeIf { it.isNotEmpty() && it != "null" }?.let { r.ip = it }
        r.port = info.optInt("port", r.port)
        token?.takeIf { it.isNotEmpty() && it != "null" }?.let { r.token = it }
        r.lastSeen = System.currentTimeMillis()
        store.save(r)
        store.currentId = id
    }

    // ------------------------------------------------------------------ messages

    private fun request(msg: JSONObject, key: String?, jsId: Int = 0, gen: Int = 0, ttl: Long = 30_000, internal: ((JSONObject) -> Unit)? = null) {
        val id = nextId++
        msg.put("id", id)
        waiting[id] = Waiter(jsId, gen, SystemClock.elapsedRealtime() + ttl, internal)
        val chunk = (minOf(mtu - 3, 512) - 2).coerceAtLeast(18)
        seq = (seq + 1) and 0xFF
        // A newer joystick sample replaces one still waiting to be sent (the robot drops a cut-off message).
        if (key != null) ops.removeAll { it is Op.Write && it.key == key }
        Protocol.encode(msg.toString(), seq, chunk).forEach { ops.addLast(Op.Write(it, key)) }
        pump()
    }

    private fun onMessage(msg: JSONObject) {
        val type = msg.optString("t")
        if (type == "evt") {
            val event = msg.optJSONObject("e") ?: return
            if (event.optString("type") == "wifi") onWifi(event)
            if (eventsWanted) listener?.onEvent(event.toString())
            return
        }
        val client = msg.optString("c")
        if (client.isNotEmpty() && client != store.clientId) return
        when {
            type == "auth" && msg.optBoolean("ok") -> {
                authed = true
                remember(msg.optString("token"))
                publish()
            }
            type == "res" && msg.optInt("s") == 401 && authed -> {
                authed = false
                publish()
            }
        }
        val w = waiting.remove(msg.optInt("re", -1)) ?: return
        val internal = w.internal
        if (internal != null) internal(msg)
        else if (w.generation == generation) listener?.onReply(w.jsId, msg.toString())
    }

    private fun onWifi(event: JSONObject) {
        val r = robot ?: JSONObject()
        val connected = event.optBoolean("connected")
        setup = !connected
        event.optString("ip").takeIf { it.isNotEmpty() && it != "null" }?.let { r.put("ip", it) }
        event.optString("ssid").takeIf { it.isNotEmpty() && it != "null" }?.let { r.put("ssid", it) }
        r.put("setup", setup)
        robot = r
        remember(null)
        publish()
    }

    private fun failWaiting(reason: String) {
        val gen = generation
        val failed = waiting.values.filter { it.internal == null && it.generation == gen }
        waiting.clear()
        val text = JSONObject().put("error", reason).toString()
        failed.forEach { listener?.onReply(it.jsId, text) }
    }

    private val housekeeping = object : Runnable {
        override fun run() {
            val now = SystemClock.elapsedRealtime()
            waiting.entries.removeAll { it.value.expires < now }
            handler.postDelayed(this, 5000)
        }
    }.also { handler.postDelayed(it, 5000) }

    // ------------------------------------------------------------------ GATT operations, one at a time

    private val opTimeout = Runnable { lost("机器人的蓝牙没有响应") }

    private fun enqueue(op: Op) {
        ops.addLast(op)
        pump()
    }

    private fun pump() {
        if (current != null) return
        val g = gatt ?: return
        val op = ops.removeFirstOrNull() ?: return
        current = op
        val started = try {
            when (op) {
                is Op.Read -> g.readCharacteristic(op.c)
                is Op.Notify -> enableNotify(g, op.c)
                is Op.Write -> writeRx(g, op.data)
            }
        } catch (e: SecurityException) {
            false
        }
        if (started) {
            busyRetries = 0
            handler.postDelayed(opTimeout, 6000)
            return
        }
        current = null
        when (op) {
            is Op.Read -> op.done(null)
            is Op.Notify -> op.done(false)
            is Op.Write -> {
                // The stack's queue is full: try again shortly.
                if (++busyRetries > 300) return lost("蓝牙发送失败")
                ops.addFirst(op)
                handler.postDelayed({ pump() }, 10)
                return
            }
        }
        pump()
    }

    private fun finish(value: ByteArray?, ok: Boolean) {
        val op = current ?: return
        handler.removeCallbacks(opTimeout)
        current = null
        when (op) {
            is Op.Read -> op.done(if (ok) value else null)
            is Op.Notify -> op.done(ok)
            is Op.Write -> Unit
        }
        pump()
    }

    @Suppress("DEPRECATION")
    private fun enableNotify(g: BluetoothGatt, c: BluetoothGattCharacteristic): Boolean {
        if (!g.setCharacteristicNotification(c, true)) return false
        val d = c.getDescriptor(Protocol.CCCD) ?: return false
        val value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
        return if (Build.VERSION.SDK_INT >= 33) g.writeDescriptor(d, value) == BluetoothStatusCodes.SUCCESS
        else {
            d.value = value
            g.writeDescriptor(d)
        }
    }

    @Suppress("DEPRECATION")
    private fun writeRx(g: BluetoothGatt, data: ByteArray): Boolean {
        val c = rx ?: return false
        val type = if (c.properties and BluetoothGattCharacteristic.PROPERTY_WRITE_NO_RESPONSE != 0)
            BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE else BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT
        return if (Build.VERSION.SDK_INT >= 33) g.writeCharacteristic(c, data, type) == BluetoothStatusCodes.SUCCESS
        else {
            c.writeType = type
            c.value = data
            g.writeCharacteristic(c)
        }
    }

    // ------------------------------------------------------------------ scanning

    private val scanTimeout = Runnable {
        stopScan()
        publish()
    }

    private fun startScan() {
        val a = adapter
        if (a == null || !a.isEnabled || !hasPermissions()) return
        if (Build.VERSION.SDK_INT <= 30 && !locationOn()) {
            error = "Android 11 及以下搜索蓝牙设备需要打开手机的「定位」开关"
            return
        }
        val scanner = a.bluetoothLeScanner ?: return
        if (scanning) stopScan()
        found.clear()
        listener?.onScan("[]")
        val filters = listOf(ScanFilter.Builder().setServiceUuid(ParcelUuid(Protocol.SERVICE)).build())
        val settings = ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build()
        try {
            scanner.startScan(filters, settings, scanCallback)
            scanning = true
            if (phase == Phase.IDLE) error = null
        } catch (e: Exception) {
            error = "搜索失败：${e.message}"
        }
        handler.postDelayed(scanTimeout, 20_000)
    }

    private fun stopScan() {
        handler.removeCallbacks(scanTimeout)
        if (!scanning) return
        scanning = false
        try {
            adapter?.bluetoothLeScanner?.stopScan(scanCallback)
        } catch (e: Exception) {
            Log.w(TAG, "stopScan: $e")
        }
    }

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            handler.post { addFound(result) }
        }

        override fun onBatchScanResults(results: MutableList<ScanResult>) {
            handler.post { results.forEach(::addFound) }
        }

        override fun onScanFailed(errorCode: Int) {
            handler.post {
                scanning = false
                error = "搜索失败（错误 $errorCode），请关掉再打开手机蓝牙"
                publish()
            }
        }
    }

    private fun addFound(r: ScanResult) {
        if (!scanning) return
        val address = r.device.address
        val saved = store.byAddress(address)
        val name = r.scanRecord?.deviceName ?: saved?.name ?: "小小白"
        found[address] = JSONObject().put("address", address).put("name", name).put("rssi", r.rssi)
            .put("saved", saved?.toJson() ?: JSONObject.NULL)
        if (!scanPosted) {
            scanPosted = true
            handler.postDelayed({
                scanPosted = false
                listener?.onScan(JSONArray(found.values.sortedByDescending { it.optInt("rssi") }).toString())
            }, 300)
        }
    }

    // ------------------------------------------------------------------ state

    private fun stateName(): String {
        val a = adapter
        return when {
            a == null -> "off"
            !hasPermissions() -> "denied"
            !a.isEnabled -> "off"
            phase == Phase.LINKED && authed -> "ready"
            phase == Phase.LINKED -> "pairing"
            phase != Phase.IDLE || retryPending -> "connecting"
            scanning -> "scanning"
            else -> "idle"
        }
    }

    private fun publish() {
        val state = stateName()
        val err = error ?: when (state) {
            "off" -> if (adapter == null) "这台手机没有蓝牙" else "手机蓝牙已关闭"
            "denied" -> "需要允许「附近的设备」权限才能连接机器人"
            else -> null
        }
        val o = JSONObject()
            .put("state", state)
            .put("error", err ?: JSONObject.NULL)
            .put("auth", authed)
            .put("setup", setup)
            .put("scanning", scanning)
            .put("address", target ?: JSONObject.NULL)
            .put("mtu", mtu)
            .put("robot", robot ?: JSONObject.NULL)
        linked = phase == Phase.LINKED
        ready = state == "ready"
        val json = o.toString()
        if (json != statusJson) {
            statusJson = json
            listener?.onState(json)
        }
    }

    private fun deviceName(): String =
        Settings.Global.getString(context.contentResolver, Settings.Global.DEVICE_NAME)?.takeIf { it.isNotBlank() } ?: Build.MODEL

    private fun locationOn(): Boolean {
        val lm = context.getSystemService(LocationManager::class.java) ?: return true
        return if (Build.VERSION.SDK_INT >= 28) lm.isLocationEnabled
        else lm.isProviderEnabled(LocationManager.GPS_PROVIDER) || lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)
    }

    companion object {
        private const val TAG = "xxb-ble"
        private const val VELOCITY = "/mc/locomotion/velocity"
        private val OPEN_PATHS = setOf("/mc/estop", "/meta/ping")

        val PERMISSIONS: Array<String> = if (Build.VERSION.SDK_INT >= 31)
            arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
        else arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
    }
}
