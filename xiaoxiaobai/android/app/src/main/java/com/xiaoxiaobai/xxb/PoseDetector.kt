package com.xiaoxiaobai.xxb

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * YOLO11-pose at 320x320 ([model] in assets/models, exported with Ultralytics: one "person" class
 * and 17 COCO keypoints, output [1, 56, 2100] in input pixels). Not thread-safe: one caller at a time.
 */
class PoseDetector(context: Context, val model: String = MODEL) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val boxed = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(boxed)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels = IntArray(SIZE * SIZE)
    private val input: FloatBuffer =
        ByteBuffer.allocateDirect(3 * SIZE * SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private var output = FloatArray(0)

    init {
        val bytes = context.assets.open(model).use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.first()
    }

    /** People in [src], coordinates in 0..1 of the picture, best first. */
    fun detect(src: Bitmap): List<Person> {
        val scale = min(SIZE / src.width.toFloat(), SIZE / src.height.toFloat())
        val padX = (SIZE - src.width * scale) / 2f
        val padY = (SIZE - src.height * scale) / 2f
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(src, Matrix().apply { setScale(scale, scale); postTranslate(padX, padY) }, paint)
        boxed.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        input.rewind()
        for (i in 0 until plane) input.put(i, ((pixels[i] shr 16) and 0xFF) / 255f)
        for (i in 0 until plane) input.put(plane + i, ((pixels[i] shr 8) and 0xFF) / 255f)
        for (i in 0 until plane) input.put(2 * plane + i, (pixels[i] and 0xFF) / 255f)

        OnnxTensor.createTensor(env, input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val out = result[0] as OnnxTensor
                val shape = out.info.shape
                val buf = out.floatBuffer
                if (output.size != buf.remaining()) output = FloatArray(buf.remaining())
                buf.get(output)
                return decode(output, shape[1].toInt(), shape[2].toInt(), scale, padX, padY, src.width, src.height)
            }
        }
    }

    private fun decode(d: FloatArray, channels: Int, anchors: Int, scale: Float, padX: Float, padY: Float, w: Int, h: Int): List<Person> {
        if (channels < 56) return emptyList()
        val found = ArrayList<Person>()
        for (i in 0 until anchors) {
            val score = d[4 * anchors + i]
            if (score < SCORE) continue
            val cx = d[i]
            val cy = d[anchors + i]
            val bw = d[2 * anchors + i]
            val bh = d[3 * anchors + i]
            val kpts = FloatArray(51)
            for (k in 0 until 17) {
                kpts[k * 3] = ((d[(5 + k * 3) * anchors + i] - padX) / scale) / w
                kpts[k * 3 + 1] = ((d[(6 + k * 3) * anchors + i] - padY) / scale) / h
                kpts[k * 3 + 2] = d[(7 + k * 3) * anchors + i]
            }
            found.add(
                Person(
                    ((cx - bw / 2f - padX) / scale) / w, ((cy - bh / 2f - padY) / scale) / h,
                    ((cx + bw / 2f - padX) / scale) / w, ((cy + bh / 2f - padY) / scale) / h,
                    score, kpts,
                ),
            )
        }
        found.sortByDescending { it.score }
        val kept = ArrayList<Person>()
        for (p in found) {
            if (kept.none { iou(it, p) > NMS_IOU }) kept.add(p)
            if (kept.size >= MAX_PEOPLE) break
        }
        return kept
    }

    private fun iou(a: Person, b: Person): Float {
        val iw = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val ih = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = iw * ih
        val union = a.area + b.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    override fun close() {
        session.close()
        boxed.recycle()
    }

    companion object {
        /** More accurate on partly seen legs; [MODEL_FAST] when the phone cannot keep up with it. */
        const val MODEL = "models/yolo11s-pose-320.onnx"
        const val MODEL_FAST = "models/yolo11n-pose-320.onnx"
        private const val SIZE = 320
        private const val SCORE = 0.25f  // enough to keep a track; starting one takes NEW_MIN_SCORE
        private const val NMS_IOU = 0.5f
        private const val MAX_PEOPLE = 5
    }
}
