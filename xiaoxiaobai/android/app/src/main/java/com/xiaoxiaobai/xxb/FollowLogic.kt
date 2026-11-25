package com.xiaoxiaobai.xxb

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.tan

/**
 * What the follow mode does with a pose detection, kept free of Android classes so it runs in
 * the JVM tests: which part of the legs the low head camera sees, which person to follow, and
 * the commands (fractions of the robot's full speed, as /mc/locomotion/velocity takes, and the
 * head's yaw in degrees).
 */

/** How much of the followed person's legs is in the picture, far to near. */
enum class Legs(val label: String) {
    FULL("整条腿"),
    LOWER("只有小腿"),
    FEET("只有脚"),
    NONE("看不到腿"),
}

/** One detected person: box corners and 17 COCO keypoints (x, y, confidence), all in 0..1 of the picture. */
class Person(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float, val kpts: FloatArray) {
    val area get() = (x2 - x1) * (y2 - y1)

    private fun seen(i: Int) = kpts.size >= 51 && kpts[i * 3 + 2] >= KPT_CONF &&
        kpts[i * 3] in EDGE..1f - EDGE && kpts[i * 3 + 1] in EDGE..1f - EDGE

    private fun y(i: Int) = kpts[i * 3 + 1]

    /** Hip, knee and ankle of one side seen top to bottom: FULL; knee above ankle: LOWER; an ankle: FEET. */
    val legs: Legs
        get() = when {
            SIDES.any { (hip, knee, ankle) -> seen(hip) && seen(knee) && seen(ankle) && y(hip) < y(knee) && y(knee) < y(ankle) } -> Legs.FULL
            SIDES.any { (_, knee, ankle) -> seen(knee) && seen(ankle) && y(knee) < y(ankle) } -> Legs.LOWER
            SIDES.any { (_, _, ankle) -> seen(ankle) } -> Legs.FEET
            else -> Legs.NONE
        }

    /**
     * Horizontal middle of one leg's visible hip, knee and ankle: of the two legs, the one nearer
     * [near] (0..1 of the picture), so the robot heads for a leg and not the gap between them.
     * The box's middle when no leg point is seen.
     */
    fun legX(near: Float = 0.5f): Float {
        val xs = SIDES.mapNotNull { side ->
            side.toList().filter { seen(it) }.takeIf { it.isNotEmpty() }?.map { kpts[it * 3] }?.average()?.toFloat()
        }
        return xs.minByOrNull { abs(it - near) } ?: ((x1 + x2) / 2f)
    }

    companion object {
        const val KPT_CONF = 0.4f
        const val EDGE = 0.01f
        private val SIDES = listOf(Triple(11, 13, 15), Triple(12, 14, 16))
    }
}

/**
 * The person to follow. [expectedX] (0..1 of the picture) is where the one followed so far
 * should be now: the nearest to it wins while close enough. Otherwise a new one is picked,
 * the biggest, and only when the detector is sure of it ([NEW_MIN_SCORE]): a low score is
 * enough to keep a track, not to start one on a chair leg.
 */
fun pickTarget(people: List<Person>, expectedX: Float?): Person? {
    if (people.isEmpty()) return null
    if (expectedX != null) {
        val near = people.minBy { abs(it.legX(expectedX) - expectedX) }
        if (abs(near.legX(expectedX) - expectedX) < TRACK_MAX_JUMP) return near
    }
    return people.filter { it.score >= NEW_MIN_SCORE }.maxByOrNull { it.area }
}

private const val TRACK_MAX_JUMP = 0.3f
const val NEW_MIN_SCORE = 0.45f

/** A detection handed to the controller: x is -1 (left edge of the picture) .. 1 (right edge), atMs when the frame was taken. */
data class Sighting(val x: Float, val legs: Legs, val atMs: Long)

/** headYaw: where the head should look [deg, + left]. */
data class FollowCommand(val vx: Float, val vtheta: Float, val headYaw: Float, val targetVx: Float, val targetVtheta: Float)

/** What the controller is doing, for the page. APPROACH and HUG only in 跟屁虫 ([FollowController.cling]). */
enum class FollowMode { TRACK, LOOK, SEARCH, SCAN, STOP, APPROACH, HUG }

/** Mean and standard deviation of the brightness (0..255) of ARGB pixels. */
fun lumaStats(pixels: IntArray): Pair<Float, Float> {
    if (pixels.isEmpty()) return 0f to 0f
    var sum = 0.0
    var sq = 0.0
    for (p in pixels) {
        val y = 0.299 * ((p shr 16) and 0xff) + 0.587 * ((p shr 8) and 0xff) + 0.114 * (p and 0xff)
        sum += y
        sq += y * y
    }
    val mean = sum / pixels.size
    val std = kotlin.math.sqrt(max(0.0, sq / pixels.size - mean * mean))
    return mean.toFloat() to std.toFloat()
}

/**
 * Whether something is pressed against the camera: the picture nearly one colour (a trouser
 * leg out of focus a few centimetres away) or dark, for [COVER_MS] in a row. Fed every analysed
 * frame; times are the frames' (elapsedRealtime ms).
 */
class CoverWatch {
    private var coveredSince = -1L
    /** When a frame was last covered (-1: never). */
    var lastCovered = -1L
        private set
    private var lastFrame = -1L

    fun frame(mean: Float, std: Float, atMs: Long) {
        lastFrame = atMs
        if (std < COVER_STD || mean < DARK_MEAN) {
            if (coveredSince < 0) coveredSince = atMs
            lastCovered = atMs
        } else {
            coveredSince = -1L
        }
    }

    /** Covered long enough to count (and frames still coming). */
    fun covered(nowMs: Long) = coveredSince >= 0 && lastFrame - coveredSince >= COVER_MS && nowMs - lastFrame < STALE_MS

    /** How long the picture has been clear [ms] (0 while covered or with no frames coming). */
    fun clearFor(nowMs: Long): Long = when {
        coveredSince >= 0 || lastFrame < 0 || nowMs - lastFrame >= STALE_MS -> 0L
        lastCovered < 0 -> nowMs
        else -> nowMs - lastCovered
    }

    fun reset() {
        coveredSince = -1L
        lastCovered = -1L
    }

    companion object {
        const val COVER_STD = 14f
        const val DARK_MEAN = 22f
        const val COVER_MS = 400L
        const val STALE_MS = 1000L
    }
}

/** Angle of a picture x (-1 left edge .. 1 right edge) off the camera's axis [deg, + left]. */
fun cameraAngle(x: Float): Float = -deg(atan(x * TAN_HALF_FOV))

/** The picture x (-1 .. 1) an angle off the camera's axis shows at [deg, + left]. */
fun pictureX(angle: Float): Float = -tan(rad(angle)) / TAN_HALF_FOV

private fun deg(r: Float) = r * 180f / PI.toFloat()
private fun rad(d: Float) = d * PI.toFloat() / 180f
private val TAN_HALF_FOV = tan(rad(FollowController.CAMERA_HFOV_DEG / 2f))

/** Into -180..180. */
private fun wrap(d: Float): Float {
    var a = d % 360f
    if (a > 180f) a -= 360f
    if (a < -180f) a += 360f
    return a
}

/**
 * 跟屁虫 lets go of the leg once the camera has stayed clear for a while ([clearForMs]): soon
 * when the person was seen walking off since ([seenAway], their legs in view again), later
 * when nobody was. The hug's own head moves uncover the camera for moments, never for this long.
 */
fun hugOver(clearForMs: Long, seenAway: Boolean) = clearForMs >= if (seenAway) RELEASE_SEEN_MS else RELEASE_ALONE_MS

const val RELEASE_SEEN_MS = 1500L
const val RELEASE_ALONE_MS = 8000L

/**
 * Turns sightings into smooth commands for the legs and the head.
 *
 * The camera sits in the head, which turns on the body, which turns on the floor; to aim one
 * at the other the controller keeps, in degrees with + to the left:
 *  - the body's heading, integrated from the turn rate it asked for (the robot's own ramp
 *    modelled, moves/walk.py), and the head's yaw on the body (the robot's aimed head move
 *    modelled, gestures.py HeadIdle with aim), both a little behind the commands;
 *  - the person's heading in that same frame: body heading + head yaw when the frame was
 *    taken + the person's angle in the picture.
 * The person's bearing from the body is then their heading minus the body's. The head looks
 * straight at that bearing, the body turns towards it, and as the body comes round the head
 * turns back towards the front by the same amount, keeping the person in the middle of the
 * picture. Each frame corrects whatever the models got wrong.
 *
 * Forward speed follows the leg view (fast with the whole leg, slow with only the lower leg,
 * stop with only the feet) through a jerk-limited ramp, only while the person is roughly
 * ahead. When nobody is seen the robot first looks where the person went (LOOK), then turns
 * on the spot that way with the head leading the turn (SEARCH), pausing after each full
 * turn for a look left and right (SCAN), until someone is found.
 *
 * 跟屁虫 ([cling]) does not stop close by: with only the feet in view it creeps on, and as soon
 * as the legs can no longer be made out (nobody detected, or a person without leg points: a leg
 * fills the picture) it keeps creeping towards where that leg was (APPROACH), unless they were
 * last seen at the edge of the picture (walking out of it). The service hugs once the camera is
 * covered or it has crept on for [CREEP_HUG_MS] ([hugging]: HUG, standing still).
 */
class FollowController(var fast: Float = 0.6f, var slow: Float = 0.3f, var cling: Boolean = false) {
    var vx = 0f
        private set
    var vtheta = 0f
        private set
    var headYaw = 0f
        private set
    /** The debounced leg view the speed is chosen from (null: nobody in view). */
    var legs: Legs? = null
        private set
    /** Starts with a look left and right before turning round. */
    var mode = FollowMode.SCAN
        private set
    /** Search direction: 1 to the left, -1 to the right. */
    var searchDir = 1f
        private set
    /** Ramp down to a stop and stay there (the user stopped following). */
    var stopping = false
    /** The robot turns the head for us (the last velocity reply carried head_yaw): otherwise it looks ahead. */
    var headSupported = false
    /** Hugging the person's leg (跟屁虫): stand still, the head is the hug's. */
    var hugging = false

    // Models of where the body points and where the head looks.
    var bodyHeading = 0f
        private set
    private var bodyRate = 0f      // deg/s
    var headPos = 0f
        private set
    private var headVel = 0f
    private val past = ArrayDeque<FloatArray>()  // (t ms since start, bodyHeading, headPos)
    private var t0 = -1L
    private val sent = ArrayDeque<FloatArray>()  // (t, vx, vtheta, headYaw) commands not applied yet

    // The person.
    private var targetHeading: Float? = null
    private var targetRate = 0f    // deg/s, how fast they move round the robot
    private var last: Sighting? = null
    private var legsAt: Long? = null   // when a leg was last made out (any leg view)
    private var legsX = 0f             // ... and where in the picture (-1 .. 1)
    private var approachSince: Long? = null
    private var lostBearing = 0f

    private var searchFrom = 0f
    private var scanSince = -1L
    private var ax = 0f
    private var candidate: Legs? = null
    private var candidateCount = 0

    /** Bearing of the person from the body [deg, + left], null before anyone was seen. */
    val bearing: Float? get() = targetHeading?.let { wrap(it - bodyHeading) }

    /** One detection result (null: nobody in this frame). */
    fun see(s: Sighting?) {
        if (s == null) return
        val (body, head) = poseAt(s.atMs)
        val heading = body + head + cameraAngle(s.x)
        val prev = last
        val fresh = prev != null && s.atMs - prev.atMs < LOST_MS
        val old = targetHeading
        // Up close a leg fills the picture: no leg points, and the box spans both legs. Keep
        // heading for the leg seen last.
        val keepHeading = s.legs == Legs.NONE && legsRecently(s.atMs, APPROACH_MS)
        if (!keepHeading) {
            if (fresh && old != null && s.atMs - prev!!.atMs >= 30) {
                val inst = wrap(heading - old) * 1000f / (s.atMs - prev.atMs)
                targetRate = (0.7f * targetRate + 0.3f * inst).coerceIn(-RATE_MAX, RATE_MAX)
            } else if (!fresh) {
                targetRate = 0f
            }
            targetHeading = heading
        }
        if (s.legs != Legs.NONE) {
            legsAt = s.atMs
            legsX = s.x
        }
        last = s
        when {
            // Too close counts at once; other changes wait for a few frames in a row.
            !fresh || s.legs == Legs.FEET -> {
                legs = s.legs
                candidate = s.legs
                candidateCount = DEBOUNCE_FRAMES
            }
            s.legs == candidate -> if (++candidateCount >= DEBOUNCE_FRAMES) legs = s.legs
            else -> {
                candidate = s.legs
                candidateCount = 1
            }
        }
    }

    fun inView(nowMs: Long) = last?.let { nowMs - it.atMs < LOST_MS } ?: false

    /**
     * A leg of the person was made out within [withinMs], after [afterMs], last away from the
     * edges of the picture (not walking out of it). Frames without leg points do not count.
     */
    fun legsRecently(nowMs: Long, withinMs: Long, afterMs: Long = Long.MIN_VALUE): Boolean {
        val at = legsAt ?: return false
        return nowMs - at < withinMs && at > afterMs && abs(legsX) < LOST_EDGE_X
    }

    /** How long it has been creeping on towards the lost leg [ms] (0: not doing that). */
    fun creptMs(nowMs: Long): Long = approachSince?.let { nowMs - it } ?: 0L

    /**
     * Where the person followed so far should show in a frame taken at [atMs] (0..1 of the
     * picture), null when they are not being tracked or should be out of the picture.
     */
    fun expectedX(atMs: Long): Float? {
        val s = last ?: return null
        val heading = targetHeading ?: return null
        if (atMs - s.atMs > TRACK_KEEP_MS) return null
        val (body, head) = poseAt(atMs)
        val ahead = heading + (targetRate * (atMs - s.atMs) / 1000f).coerceIn(-30f, 30f)
        val angle = wrap(ahead - body - head)
        if (abs(angle) > CAMERA_HFOV_DEG / 2f + 5f) return null
        return (pictureX(angle) + 1f) / 2f
    }

    /** Advance by dt seconds and return the command to send. */
    fun tick(nowMs: Long, dt: Float): FollowCommand {
        advanceModels(nowMs, dt)
        val s = last.takeIf { !stopping && inView(nowMs) }
        if (s == null) legs = null
        var targetVx = 0f
        var turnRate = 0f  // deg/s wanted from the body
        var head = 0f
        val heading = targetHeading
        val legsLost = s == null || (legs ?: s.legs) == Legs.NONE
        when {
            stopping -> mode = FollowMode.STOP
            hugging -> mode = FollowMode.HUG
            cling && heading != null && legsLost && legsRecently(nowMs, APPROACH_MS) -> {
                // The legs lost in the middle of the picture: a leg fills it. Creep on towards where it was.
                mode = FollowMode.APPROACH
                val b = wrap(heading - bodyHeading)
                head = b
                turnRate = TURN_K * if (abs(b) < TURN_DEADBAND) 0f else b
                targetVx = CLING_VX * facing(b)
                searchFrom = bodyHeading
            }
            s != null && heading != null -> {
                mode = FollowMode.TRACK
                val b = wrap(heading - bodyHeading)
                lostBearing = b
                if (abs(b) > 4f) searchDir = sign(b)
                head = b
                turnRate = TURN_K * if (abs(b) < TURN_DEADBAND) 0f else b
                targetVx = when (legs ?: s.legs) {
                    Legs.FULL -> fast
                    Legs.LOWER -> slow
                    Legs.FEET -> if (cling) CLING_VX else 0f
                    Legs.NONE -> 0f
                }
                // Off to a side: turn first, walk once roughly facing them.
                targetVx *= facing(b)
            }
            heading != null && nowMs - last!!.atMs < LOOK_MS -> {
                // Just lost: look where they were heading, the body following.
                mode = FollowMode.LOOK
                if (abs(targetRate) > 12f) searchDir = sign(targetRate)
                val lostFor = (nowMs - last!!.atMs) / 1000f
                val ahead = heading + (targetRate * lostFor).coerceIn(-45f, 45f) + searchDir * LOOK_LEAD_DEG
                val b = wrap(ahead - bodyHeading)
                head = b
                turnRate = TURN_K * b
                searchFrom = bodyHeading
            }
            else -> {
                if (mode != FollowMode.SEARCH && mode != FollowMode.SCAN) {
                    mode = FollowMode.SEARCH
                    searchFrom = bodyHeading
                }
                if (mode == FollowMode.SEARCH && abs(bodyHeading - searchFrom) >= 360f) {
                    mode = FollowMode.SCAN
                    scanSince = nowMs
                }
                if (mode == FollowMode.SCAN) {
                    if (scanSince < 0) scanSince = nowMs
                    val since = (nowMs - scanSince) / 1000f
                    head = searchDir * HEAD_MAX * sin(2f * PI.toFloat() * since / SCAN_S)
                    if (since >= SCAN_S) {
                        mode = FollowMode.SEARCH
                        searchFrom = bodyHeading
                    }
                } else {
                    head = searchDir * SEARCH_HEAD_LEAD
                    turnRate = searchDir * SEARCH_RATE
                }
            }
        }
        if (mode != FollowMode.APPROACH) approachSince = null else if (approachSince == null) approachSince = nowMs
        if (!headSupported) head = 0f
        headYaw = head.coerceIn(-HEAD_MAX, HEAD_MAX)
        // Turning in place the robot turns up to twice as fast as while walking (input_source.py scale_velocity).
        val walking = targetVx > 0f || vx > 0.005f
        val targetVtheta = (turnRate / if (walking) TURN_MAX_MOVING else TURN_MAX_STILL).coerceIn(-TURN_MAX, TURN_MAX)
        val close = s != null && (legs ?: s.legs) == Legs.FEET
        rampVx(targetVx, dt, if (close) DECEL_CLOSE else DECEL)
        val step = TURN_RATE * dt
        vtheta += (targetVtheta - vtheta).coerceIn(-step, step)
        if (abs(vtheta) < 1e-4f && targetVtheta == 0f) vtheta = 0f
        sent.addLast(floatArrayOf((nowMs - t0).toFloat(), vx, vtheta, headYaw))
        return FollowCommand(vx, vtheta, headYaw, targetVx, targetVtheta)
    }

    /** 1 with the person within [AHEAD_DEG] of straight ahead, down to 0 past [SIDE_DEG]. */
    private fun facing(bearing: Float) = 1f - ((abs(bearing) - AHEAD_DEG) / (SIDE_DEG - AHEAD_DEG)).coerceIn(0f, 1f)

    /** Stopped: no forward speed left and no turn. */
    fun still() = vx < 0.005f && abs(vtheta) < 0.005f

    /** Back to standing, e.g. when the robot refused the command or someone else took over. */
    fun reset() {
        vx = 0f
        ax = 0f
        vtheta = 0f
        bodyRate = 0f
        sent.clear()
        applied = floatArrayOf(0f, 0f, 0f, headYaw)
    }

    // ------------------------------------------------------------------ models

    /** The body heading and head yaw at [atMs] (interpolated from the recent past). */
    private fun poseAt(atMs: Long): Pair<Float, Float> {
        if (past.isEmpty() || t0 < 0) return bodyHeading to headPos
        val t = (atMs - t0).toFloat()
        if (t >= past.last()[0]) return past.last()[1] to past.last()[2]
        if (t <= past.first()[0]) return past.first()[1] to past.first()[2]
        val i = past.indexOfFirst { it[0] >= t }
        val a = past[i - 1]
        val b = past[i]
        val f = (t - a[0]) / (b[0] - a[0]).coerceAtLeast(1e-3f)
        return (a[1] + (b[1] - a[1]) * f) to (a[2] + (b[2] - a[2]) * f)
    }

    private fun advanceModels(nowMs: Long, dt: Float) {
        if (t0 < 0) t0 = nowMs
        val t = (nowMs - t0).toFloat()
        // The robot acts on a command some time after it was sent: the API writes it to the
        // control loop every 100 ms, the loop runs at 50 Hz.
        var cmd: FloatArray? = null
        while (sent.isNotEmpty() && sent.first()[0] <= t - COMMAND_LAG_MS) cmd = sent.removeFirst()
        if (cmd != null) applied = cmd
        val (cvx, cvtheta, chead) = applied.let { Triple(it[1], it[2], it[3]) }
        val maxRate = if (cvx > 1e-6f) TURN_MAX_MOVING else TURN_MAX_STILL
        val wantRate = cvtheta * maxRate
        var left = dt
        while (left > 1e-6f) {
            val h = min(left, 0.01f)
            left -= h
            val slowing = abs(wantRate) < abs(bodyRate) || wantRate * bodyRate < 0f
            val acc = (if (slowing) TURN_MAX_STILL / ROBOT_RAMP_DOWN_S else TURN_MAX_STILL / ROBOT_RAMP_UP_S) * h
            bodyRate += (wantRate - bodyRate).coerceIn(-acc, acc)
            bodyHeading += bodyRate * h
            val aim = if (headSupported) chead.coerceIn(-HEAD_MAX, HEAD_MAX) else 0f
            val a = HEAD_OMEGA * HEAD_OMEGA * (aim - headPos) - 2f * HEAD_OMEGA * headVel
            headVel = (headVel + a * h).coerceIn(-HEAD_MAX_SPEED, HEAD_MAX_SPEED)
            headPos += headVel * h
        }
        past.addLast(floatArrayOf(t, bodyHeading, headPos))
        while (past.size > 2 && past.first()[0] < t - HISTORY_MS) past.removeFirst()
    }

    private var applied = floatArrayOf(0f, 0f, 0f, 0f)

    // The acceleration moves towards K * error, bounded by the accel / decel limits, at most
    // JERK per second: an S-shaped start and a soft landing on the target speed.
    private fun rampVx(target: Float, dt: Float, decel: Float) {
        val want = (K * (target - vx)).coerceIn(-decel, ACCEL)
        val jerk = (if (want < ax) max(JERK, decel * 2f) else JERK) * dt
        ax += (want - ax).coerceIn(-jerk, jerk)
        val before = vx
        vx = max(0f, vx + ax * dt)
        if ((target - before) * (target - vx) < 0f) {
            vx = target
            ax = 0f
        }
        if (target == 0f && vx < 0.002f) {
            vx = 0f
            ax = 0f
        }
    }

    companion object {
        const val LOST_MS = 700L
        const val TRACK_KEEP_MS = 1500L
        const val LOOK_MS = LOST_MS + 1500L
        const val DEBOUNCE_FRAMES = 3
        const val ACCEL = 0.2f        // full speed fraction per second
        const val DECEL = 0.3f
        const val DECEL_CLOSE = 0.6f  // only the feet in view: stop sooner
        const val JERK = 0.6f
        const val K = 2.5f
        const val TURN_K = 1.6f       // body turn [deg/s] per degree of bearing
        const val TURN_MAX = 0.6f
        const val TURN_RATE = 1.0f    // command change per second
        const val TURN_DEADBAND = 3f  // deg: detection jitter does not wiggle the body
        const val AHEAD_DEG = 12f     // full speed while the person is within this of straight ahead
        const val SIDE_DEG = 40f      // no forward speed past this
        const val LOOK_LEAD_DEG = 10f
        const val SEARCH_RATE = 40f   // deg/s turning on the spot
        const val SEARCH_HEAD_LEAD = 30f
        const val SCAN_S = 4f
        const val RATE_MAX = 90f
        const val CLING_VX = 0.2f      // 跟屁虫 up close: creep on at this fraction of full speed
        const val APPROACH_MS = 4000L  // ... for this long after last making out their legs
        const val CREEP_HUG_MS = 1000L // crept on this long without seeing the legs: up against one, hug it
        const val LOST_EDGE_X = 0.7f   // legs last seen further out than this (-1 .. 1): walked out of the picture
        const val HISTORY_MS = 1500f

        /** Head-board camera, horizontal. */
        const val CAMERA_HFOV_DEG = 60f
        // The robot's aimed head move (gestures.py AIM_YAW_DEG / AIM_OMEGA, IDLE_MAX_SPEED_DEG_S).
        const val HEAD_MAX = 34f
        const val HEAD_OMEGA = 7f
        const val HEAD_MAX_SPEED = 160f
        // The robot's turn (constants.py VTHETA_MAX_*, VELOCITY_RAMP_*), in deg/s.
        val TURN_MAX_STILL = 1.5f * 180f / PI.toFloat()
        val TURN_MAX_MOVING = 0.75f * 180f / PI.toFloat()
        const val ROBOT_RAMP_UP_S = 1.0f
        const val ROBOT_RAMP_DOWN_S = 0.8f
        const val COMMAND_LAG_MS = 100f
    }
}