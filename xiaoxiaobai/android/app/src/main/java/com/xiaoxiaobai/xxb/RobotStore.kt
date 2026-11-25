package com.xiaoxiaobai.xxb

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/** A robot this phone knows: its Bluetooth address, last Wi-Fi address and pairing token. */
data class Robot(
    val id: String,
    var name: String,
    var address: String? = null,
    var ip: String? = null,
    var port: Int = 8000,
    var token: String? = null,
    var lastSeen: Long = 0,
) {
    fun toJson(withToken: Boolean = false): JSONObject = JSONObject().apply {
        put("id", id)
        put("name", name)
        put("address", address ?: JSONObject.NULL)
        put("ip", ip ?: JSONObject.NULL)
        put("port", port)
        put("paired", token != null)
        put("last_seen", lastSeen)
        if (withToken) put("token", token ?: JSONObject.NULL)
    }

    val url: String? get() = ip?.let { "http://$it:$port/" }

    companion object {
        fun from(o: JSONObject) = Robot(
            id = o.getString("id"),
            name = o.optString("name", "小小白"),
            address = o.optString("address").takeIf { it.isNotEmpty() && it != "null" },
            ip = o.optString("ip").takeIf { it.isNotEmpty() && it != "null" },
            port = o.optInt("port", 8000),
            token = o.optString("token").takeIf { it.isNotEmpty() && it != "null" },
            lastSeen = o.optLong("last_seen"),
        )
    }
}

/** App settings in private SharedPreferences (the tokens never leave the phone). */
class RobotStore(context: Context) {
    private val prefs = context.getSharedPreferences("xxb", Context.MODE_PRIVATE)

    val clientId: String = prefs.getString("client_id", null) ?: UUID.randomUUID().toString().also {
        prefs.edit().putString("client_id", it).apply()
    }

    @Synchronized
    fun all(): List<Robot> {
        val arr = JSONArray(prefs.getString("robots", "[]"))
        return (0 until arr.length()).map { Robot.from(arr.getJSONObject(it)) }.sortedByDescending { it.lastSeen }
    }

    fun get(id: String?): Robot? = id?.let { wanted -> all().firstOrNull { it.id == wanted } }

    fun byAddress(address: String?): Robot? = address?.let { a -> all().firstOrNull { it.address.equals(a, ignoreCase = true) } }

    @Synchronized
    fun save(robot: Robot) {
        val rest = all().filter { it.id != robot.id }
        val arr = JSONArray()
        (listOf(robot) + rest).forEach { arr.put(it.toJson(withToken = true)) }
        prefs.edit().putString("robots", arr.toString()).apply()
    }

    @Synchronized
    fun update(id: String, change: (Robot) -> Unit): Robot? {
        val robot = get(id) ?: return null
        change(robot)
        save(robot)
        return robot
    }

    @Synchronized
    fun remove(id: String) {
        val arr = JSONArray()
        all().filter { it.id != id }.forEach { arr.put(it.toJson(withToken = true)) }
        prefs.edit().putString("robots", arr.toString()).apply()
        if (currentId == id) currentId = null
    }

    var currentId: String?
        get() = prefs.getString("current", null)
        set(value) = prefs.edit().putString("current", value).apply()

    var theme: String
        get() = prefs.getString("theme", "dark") ?: "dark"
        set(value) = prefs.edit().putString("theme", value).apply()

    /** Follow mode speeds, fractions of full speed: whole leg in view (far) and lower leg only (near). */
    var followFast: Float
        get() = prefs.getFloat("follow_fast", 0.6f)
        set(value) = prefs.edit().putFloat("follow_fast", value).apply()

    var followSlow: Float
        get() = prefs.getFloat("follow_slow", 0.3f)
        set(value) = prefs.edit().putFloat("follow_slow", value).apply()
}
