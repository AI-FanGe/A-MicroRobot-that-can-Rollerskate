package com.xiaoxiaobai.xxb

import java.io.BufferedInputStream
import java.io.Closeable
import java.io.EOFException
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** The ESP32's multipart MJPEG stream (firmware/xiao_face streamHandler): one JPEG per next(). */
class MjpegReader(url: String) : Closeable {
    private val conn = (URL(url).openConnection() as HttpURLConnection).apply {
        connectTimeout = 3000
        readTimeout = 5000
        useCaches = false
    }
    private val input: BufferedInputStream

    init {
        val code = conn.responseCode
        if (code != 200) {
            conn.disconnect()
            throw IOException("摄像头回应 HTTP $code")
        }
        input = BufferedInputStream(conn.inputStream, 64 * 1024)
    }

    fun next(): ByteArray {
        var length = -1
        while (true) {
            val line = readLine()
            if (line.isEmpty()) {
                if (length >= 0) break
                continue
            }
            if (line.startsWith("Content-Length", ignoreCase = true)) {
                length = line.substringAfter(':').trim().toIntOrNull() ?: -1
            }
        }
        if (length !in 1..MAX_JPEG) throw IOException("画面大小不对：$length")
        val frame = ByteArray(length)
        var got = 0
        while (got < length) {
            val n = input.read(frame, got, length - got)
            if (n < 0) throw EOFException("摄像头断开了")
            got += n
        }
        return frame
    }

    private fun readLine(): String {
        val sb = StringBuilder()
        while (true) {
            val c = input.read()
            if (c < 0) throw EOFException("摄像头断开了")
            if (c == '\n'.code) break
            if (c != '\r'.code && sb.length < 256) sb.append(c.toChar())
        }
        return sb.toString()
    }

    override fun close() {
        runCatching { input.close() }
        conn.disconnect()
    }

    private companion object {
        const val MAX_JPEG = 256 * 1024
    }
}
