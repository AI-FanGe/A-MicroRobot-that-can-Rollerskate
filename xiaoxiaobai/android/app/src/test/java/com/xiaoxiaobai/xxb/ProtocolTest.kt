package com.xiaoxiaobai.xxb

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ProtocolTest {
    private fun hex(s: String) = ByteArray(s.length / 2) { s.substring(it * 2, it * 2 + 2).toInt(16).toByte() }

    // Packets from src/api/ble.py encode({"t": "evt", "e": {"type": "status", "note": "表情" * 60}}, 9, 20).
    private val fromRobot = listOf(
        "0509789cab562a51b2524a2d2b51d2514a55b2aa562a",
        "0009a92c00d24ac5258925a5c540c1bcfc1210ffc5c2",
        "020915cf9a5b47495a934ab5b500cf2d04a4",
    )

    @Test
    fun decodesTheRobotsCompressedMessage() {
        val a = Assembler()
        val msg = fromRobot.map { a.feed(hex(it)) }.last()!!
        assertEquals("evt", msg.getString("t"))
        assertEquals("表情".repeat(60), msg.getJSONObject("e").getString("note"))
    }

    @Test
    fun dropsAMessageWhoseStartWasLost() {
        val a = Assembler()
        assertNull(a.feed(hex(fromRobot[1])))
        assertNull(a.feed(hex(fromRobot[2])))
    }

    @Test
    fun roundTripsLargeAndSmallMessages() {
        for (text in listOf("""{"t":"req","id":1,"m":"POST","p":"/mc/estop","b":{}}""",
            JSONObject().put("t", "res").put("r", "小小白 ".repeat(400)).toString())) {
            val packets = Protocol.encode(text, 200, 180)
            assertTrue(packets.all { it.size <= 182 })
            val a = Assembler()
            val msg = packets.map { a.feed(it) }.last()!!
            assertEquals(JSONObject(text).toString(), msg.toString())
        }
    }
}
