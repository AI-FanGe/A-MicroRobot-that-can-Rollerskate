package com.xiaoxiaobai.xxb

import org.json.JSONException
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.UUID
import java.util.zip.DataFormatException
import java.util.zip.Deflater
import java.util.zip.Inflater

/**
 * The robot's Bluetooth protocol (src/api/ble.py): JSON messages cut into packets of
 * [flags, seq, bytes…]; flags bit 0 = first packet, bit 1 = last, bit 2 = zlib-compressed.
 */
object Protocol {
    val SERVICE: UUID = UUID.fromString("f3a50001-6b2c-4e5d-8f9a-786962726f74")
    val RX: UUID = UUID.fromString("f3a50002-6b2c-4e5d-8f9a-786962726f74")
    val TX: UUID = UUID.fromString("f3a50003-6b2c-4e5d-8f9a-786962726f74")
    val INFO: UUID = UUID.fromString("f3a50004-6b2c-4e5d-8f9a-786962726f74")
    val CCCD: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")

    const val FIRST = 0x01
    const val LAST = 0x02
    const val DEFLATE = 0x04
    private const val COMPRESS_OVER = 160
    const val MAX_MESSAGE = 256 * 1024

    fun encode(json: String, seq: Int, chunk: Int): List<ByteArray> {
        var raw = json.toByteArray(Charsets.UTF_8)
        var flags = 0
        if (raw.size > COMPRESS_OVER) {
            val packed = deflate(raw)
            if (packed.size < raw.size) {
                raw = packed
                flags = DEFLATE
            }
        }
        val parts = if (raw.isEmpty()) listOf(ByteArray(0))
        else (raw.indices step chunk).map { raw.copyOfRange(it, minOf(it + chunk, raw.size)) }
        return parts.mapIndexed { i, part ->
            var f = if (i == 0) flags or FIRST else 0
            if (i == parts.lastIndex) f = f or LAST
            byteArrayOf(f.toByte(), (seq and 0xFF).toByte()) + part
        }
    }

    private fun deflate(data: ByteArray): ByteArray {
        val deflater = Deflater(6)
        deflater.setInput(data)
        deflater.finish()
        val out = ByteArrayOutputStream(data.size)
        val buf = ByteArray(2048)
        while (!deflater.finished()) out.write(buf, 0, deflater.deflate(buf))
        deflater.end()
        return out.toByteArray()
    }

    fun inflate(data: ByteArray): ByteArray? {
        val inflater = Inflater()
        inflater.setInput(data)
        val out = ByteArrayOutputStream(data.size * 4)
        val buf = ByteArray(4096)
        try {
            while (!inflater.finished()) {
                val n = inflater.inflate(buf)
                if (n == 0 && (inflater.needsInput() || inflater.needsDictionary())) return null
                out.write(buf, 0, n)
                if (out.size() > MAX_MESSAGE) return null
            }
        } catch (e: DataFormatException) {
            return null
        } finally {
            inflater.end()
        }
        return out.toByteArray()
    }
}

/** Packets from the robot back into messages; a message whose first packet was lost is dropped. */
class Assembler {
    private var buf: ByteArrayOutputStream? = null
    private var seq = -1
    private var deflated = false

    fun feed(packet: ByteArray): JSONObject? {
        if (packet.size < 2) return null
        val flags = packet[0].toInt() and 0xFF
        val s = packet[1].toInt() and 0xFF
        if (flags and Protocol.FIRST != 0) {
            buf = ByteArrayOutputStream()
            seq = s
            deflated = flags and Protocol.DEFLATE != 0
        } else if (buf == null || s != seq) {
            buf = null
            return null
        }
        val b = buf ?: return null
        b.write(packet, 2, packet.size - 2)
        if (b.size() > Protocol.MAX_MESSAGE) {
            buf = null
            return null
        }
        if (flags and Protocol.LAST == 0) return null
        buf = null
        val data = if (deflated) Protocol.inflate(b.toByteArray()) ?: return null else b.toByteArray()
        return try {
            JSONObject(String(data, Charsets.UTF_8))
        } catch (e: JSONException) {
            null
        }
    }

    fun reset() {
        buf = null
    }
}
