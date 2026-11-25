package com.xiaoxiaobai.xxb

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class FollowLogicTest {
    /** A person whose given keypoints are seen at (x, y); the others are not. */
    private fun person(vararg points: Pair<Int, Pair<Float, Float>>, x1: Float = 0.3f, x2: Float = 0.5f, score: Float = 0.8f): Person {
        val k = FloatArray(51)
        for ((i, p) in points) {
            k[i * 3] = p.first
            k[i * 3 + 1] = p.second
            k[i * 3 + 2] = 0.9f
        }
        return Person(x1, 0.1f, x2, 0.95f, score, k)
    }

    private val fullLeg = person(11 to (0.4f to 0.2f), 13 to (0.4f to 0.5f), 15 to (0.4f to 0.85f))
    private val lowerLeg = person(13 to (0.4f to 0.15f), 15 to (0.4f to 0.8f))
    private val feet = person(15 to (0.4f to 0.6f), 16 to (0.5f to 0.6f))

    @Test
    fun legsFromKeypoints() {
        assertEquals(Legs.FULL, fullLeg.legs)
        assertEquals(Legs.LOWER, lowerLeg.legs)
        assertEquals(Legs.FEET, feet.legs)
        assertEquals(Legs.NONE, person(0 to (0.5f to 0.1f)).legs)
        // A hip guessed outside the picture does not count.
        assertEquals(Legs.LOWER, person(11 to (0.4f to -0.1f), 13 to (0.4f to 0.15f), 15 to (0.4f to 0.8f)).legs)
        // One leg, not the gap between them: the one nearer where it is looked for.
        assertEquals(0.5f, feet.legX(), 1e-6f)
        assertEquals(0.4f, feet.legX(0.3f), 1e-6f)
        val twoLegs = person(13 to (0.3f to 0.2f), 15 to (0.32f to 0.8f), 14 to (0.7f to 0.2f), 16 to (0.68f to 0.8f), x1 = 0.25f, x2 = 0.75f)
        assertEquals(0.31f, twoLegs.legX(0.4f), 1e-5f)
        assertEquals(0.69f, twoLegs.legX(0.6f), 1e-5f)
        assertEquals("no leg point: the box", 0.4f, person(0 to (0.5f to 0.1f)).legX(), 1e-6f)
    }

    @Test
    fun keepsFollowingTheSamePerson() {
        val left = person(15 to (0.2f to 0.8f), x1 = 0.1f, x2 = 0.3f)
        val bigRight = person(15 to (0.8f to 0.8f), x1 = 0.5f, x2 = 0.99f)
        assertSame(bigRight, pickTarget(listOf(left, bigRight), null))
        assertSame(left, pickTarget(listOf(left, bigRight), 0.25f))
        assertNull(pickTarget(emptyList(), 0.25f))
        // A shaky detection keeps a track but does not start one.
        val unsure = person(15 to (0.5f to 0.8f), score = 0.3f)
        assertSame(unsure, pickTarget(listOf(unsure), 0.45f))
        assertNull(pickTarget(listOf(unsure), null))
    }

    @Test
    fun anglesInThePicture() {
        assertEquals(0f, cameraAngle(0f), 1e-6f)
        assertEquals(FollowController.CAMERA_HFOV_DEG / 2f, cameraAngle(-1f), 1e-3f)  // left edge: + left
        assertEquals(0.5f, pictureX(cameraAngle(0.5f)), 1e-5f)
    }

    /** Run the controller at 10 Hz for [seconds], seeing [s] (moved to "now") every tick. */
    private fun FollowController.run(seconds: Float, start: Long, s: Sighting?, onTick: (FollowCommand) -> Unit = {}): Long {
        var t = start
        repeat((seconds * 10).toInt()) {
            t += 100
            if (s != null) see(s.copy(atMs = t))
            onTick(tick(t, 0.1f))
        }
        return t
    }

    /**
     * Run against a little world: the robot does exactly what the controller models, the person
     * stands at the heading [personAt] gives (null: gone), seen when inside the picture.
     */
    private fun FollowController.world(seconds: Float, start: Long, legs: Legs = Legs.FULL, personAt: (Long) -> Float?,
                                       onTick: (FollowCommand) -> Unit = {}): Long {
        var t = start
        repeat((seconds * 10).toInt()) {
            t += 100
            personAt(t)?.let { heading ->
                var angle = (heading - bodyHeading - headPos) % 360f
                if (angle > 180f) angle -= 360f
                if (angle < -180f) angle += 360f
                if (abs(angle) < FollowController.CAMERA_HFOV_DEG / 2f) see(Sighting(pictureX(angle), legs, t))
            }
            onTick(tick(t, 0.1f))
        }
        return t
    }

    @Test
    fun startsGentlyAndReachesTheFastSpeed() {
        val c = FollowController(fast = 0.6f, slow = 0.3f)
        val seen = mutableListOf<Float>()
        c.run(8f, 0, Sighting(0f, Legs.FULL, 0)) { seen.add(it.vx) }
        assertTrue("first 0.3 s: ${seen[2]}", seen[2] < 0.04f)
        assertTrue("never jumps", seen.zipWithNext().all { (a, b) -> b - a <= 0.2f * 0.1f + 1e-4f })
        assertTrue("reaches it: ${seen.last()}", seen.last() > 0.59f)
        assertTrue("no overshoot: ${seen.max()}", seen.max() <= 0.6f + 1e-4f)
    }

    @Test
    fun slowsDownForTheLowerLegAndStopsGraduallyWhenLost() {
        val c = FollowController(fast = 0.6f, slow = 0.3f)
        var t = c.run(8f, 0, Sighting(0f, Legs.FULL, 0))
        t = c.run(6f, t, Sighting(0f, Legs.LOWER, 0))
        assertEquals(0.3f, c.vx, 0.01f)

        val down = mutableListOf<Float>()
        c.run(5f, t, null) { down.add(it.vx) }
        assertTrue("holds through a missed frame or two", down[3] > 0.29f)
        assertTrue("gradual", down.zipWithNext().all { (a, b) -> b <= a && a - b <= 0.3f * 0.1f + 1e-4f })
        assertEquals(0f, down.last(), 0f)
    }

    @Test
    fun lowerLegNeedsAFewFramesButFeetStopAtOnce() {
        val c = FollowController(fast = 0.6f, slow = 0.3f)
        var t = c.run(8f, 0, Sighting(0f, Legs.FULL, 0))
        c.see(Sighting(0f, Legs.LOWER, t + 10))
        assertEquals(Legs.FULL, c.legs)
        c.see(Sighting(0f, Legs.FEET, t + 20))
        assertEquals(Legs.FEET, c.legs)
        t = c.run(4f, t + 20, Sighting(0f, Legs.FEET, 0))
        assertEquals(0f, c.vx, 0f)
    }

    @Test
    fun headLooksFirstThenTheBodyComesRound() {
        val c = FollowController().apply { headSupported = true }
        val heads = mutableListOf<Float>()
        val turns = mutableListOf<Float>()
        c.world(6f, 0, personAt = { 25f }) {
            heads.add(c.headPos)
            turns.add(it.vtheta)
        }
        assertTrue("the head turns left to the person first: ${heads.take(10)}", heads.take(10).max() > 12f)
        assertTrue("the body turns left: ${turns.take(10)}", turns.take(10).max() > 0.1f)
        // Within the turn deadband; the head makes up the last few degrees.
        assertEquals("the body ends facing them", 25f, c.bodyHeading, 4f)
        assertEquals("and the head back to the front", 0f, c.headPos, 4f)
        assertEquals("the person in the middle of the picture", 25f, c.bodyHeading + c.headPos, 1f)
    }

    @Test
    fun withoutTheHeadTheBodyStillTurns() {
        val c = FollowController()
        c.world(6f, 0, personAt = { -20f })
        assertEquals(0f, c.headPos, 0f)
        assertEquals(-20f, c.bodyHeading, 4f)
    }

    @Test
    fun walksOnlyOnceRoughlyFacingThem() {
        val c = FollowController().apply { headSupported = true }
        val early = mutableListOf<FollowCommand>()
        c.world(1f, 0, personAt = { 60f }) { early.add(it) }
        assertTrue("60 deg off: turn, no walking yet ${early.map { it.targetVx }}", early.all { it.targetVx == 0f })
        c.world(8f, 1000, personAt = { 60f })
        assertTrue("facing them: walking ${c.vx}", c.vx > 0.3f)
    }

    @Test
    fun searchesTheWayThePersonLeftUntilFound() {
        val c = FollowController().apply { headSupported = true }
        // They walk off to the right (heading decreasing) and out of reach of the picture.
        var t = c.world(3f, 0, personAt = { -15f * it / 1000f })
        t = c.world(0.5f, t, personAt = { null })
        val looks = mutableListOf<FollowCommand>()
        t = c.world(1.5f, t, personAt = { null }) { looks.add(it) }
        assertEquals(FollowMode.LOOK, c.mode)
        assertTrue("looks right: ${looks.map { it.headYaw }}", looks.last().headYaw < -10f)
        assertEquals(-1f, c.searchDir)

        t = c.world(3f, t, personAt = { null })
        assertEquals(FollowMode.SEARCH, c.mode)
        assertTrue("turning right with the head leading: ${c.vtheta}, ${c.headYaw}", c.vtheta < -0.2f && c.headYaw < -20f)

        // Someone stands well round to the right: found, followed, faced.
        val there = c.bodyHeading - 70f
        t = c.world(8f, t, personAt = { there })
        assertEquals(FollowMode.TRACK, c.mode)
        assertEquals(0f, c.bearing!!, 4f)
    }

    @Test
    fun aFullTurnThenALookAround() {
        val c = FollowController().apply { headSupported = true }
        c.world(4.5f, 0, personAt = { null })
        assertEquals("starts with a look left and right", FollowMode.SEARCH, c.mode)
        var scanned = false
        c.world(14f, 4500, personAt = { null }) { scanned = scanned || c.mode == FollowMode.SCAN }
        assertTrue("one turn round (${c.bodyHeading}), then the head scans", scanned)
    }

    @Test
    fun stoppingRampsDownAndStays() {
        val c = FollowController()
        val t = c.run(8f, 0, Sighting(0f, Legs.FULL, 0))
        c.stopping = true
        c.run(4f, t, Sighting(0f, Legs.FULL, 0))
        assertTrue(c.still())
    }

    @Test
    fun clingCreepsOnUpCloseAndThenTowardsWhereTheyWere() {
        val c = FollowController(cling = true).apply { headSupported = true }
        var t = c.run(6f, 0, Sighting(0f, Legs.LOWER, 0))
        t = c.run(4f, t, Sighting(0f, Legs.FEET, 0))
        assertEquals("only the feet: creeps on", FollowController.CLING_VX, c.vx, 0.01f)
        assertTrue("legs seen", c.legsRecently(t, 1000))

        // Their legs fill the picture: nobody detected.
        val lost = mutableListOf<FollowCommand>()
        t = c.run(2f, t, null) { lost.add(it) }
        assertEquals(FollowMode.APPROACH, c.mode)
        assertTrue("still creeping: ${lost.map { it.vx }}", lost.all { it.vx > 0.1f })
        assertTrue("long enough to hug: ${c.creptMs(t)}", c.creptMs(t) >= FollowController.CREEP_HUG_MS)
        assertTrue("legs still recent for the hug", c.legsRecently(t, FollowController.APPROACH_MS))
        c.run(3f, t, null)
        assertTrue("gives up after a while: ${c.mode}", c.mode == FollowMode.LOOK || c.mode == FollowMode.SEARCH)
        assertEquals(0L, c.creptMs(t + 3000))
    }

    @Test
    fun clingCreepsOnEvenWhenTheWholeLegWasLastSeen() {
        val c = FollowController(cling = true).apply { headSupported = true }
        var t = c.run(4f, 0, Sighting(0f, Legs.FULL, 0))
        t = c.run(2f, t, null)
        assertEquals(FollowMode.APPROACH, c.mode)
        assertTrue(c.creptMs(t) >= FollowController.CREEP_HUG_MS)
    }

    @Test
    fun clingDoesNotCreepAfterSomeoneWalkingOutOfThePicture() {
        val c = FollowController(cling = true).apply { headSupported = true }
        val t = c.run(4f, 0, Sighting(0.85f, Legs.FULL, 0))
        c.run(1.5f, t, null)
        assertEquals(FollowMode.LOOK, c.mode)
        assertEquals(0L, c.creptMs(t + 1500))
    }

    @Test
    fun clingCreepsOnWhenALegFillsThePicture() {
        val c = FollowController(cling = true).apply { headSupported = true }
        var t = c.run(4f, 0, Sighting(0.2f, Legs.FEET, 0))
        val heading = c.bodyHeading + c.bearing!!
        // A person still detected, no leg points, the box (both legs) off to the side.
        val creep = mutableListOf<FollowCommand>()
        t = c.run(2f, t, Sighting(-0.6f, Legs.NONE, 0)) { creep.add(it) }
        assertEquals(FollowMode.APPROACH, c.mode)
        assertTrue("creeping on: ${creep.takeLast(10).map { it.vx }}", creep.takeLast(10).all { it.vx > 0.1f })
        assertEquals("heading for the leg seen last", heading, c.bodyHeading + c.bearing!!, 0.5f)
        assertTrue("legs recent", c.legsRecently(t, FollowController.APPROACH_MS))
        assertTrue(c.creptMs(t) >= FollowController.CREEP_HUG_MS)

        // The whole leg in view again: back to following.
        c.run(1f, t, Sighting(0f, Legs.FULL, 0))
        assertEquals(FollowMode.TRACK, c.mode)
        assertEquals(0L, c.creptMs(t + 1000))
    }

    @Test
    fun followWithoutClingStopsWithoutLegPoints() {
        val c = FollowController()
        val t = c.run(4f, 0, Sighting(0f, Legs.LOWER, 0))
        c.run(3f, t, Sighting(0f, Legs.NONE, 0))
        assertEquals(FollowMode.TRACK, c.mode)
        assertEquals(0f, c.vx, 0f)
    }

    @Test
    fun followWithoutClingStillStopsAtTheFeet() {
        val c = FollowController()
        var t = c.run(6f, 0, Sighting(0f, Legs.LOWER, 0))
        t = c.run(4f, t, Sighting(0f, Legs.FEET, 0))
        assertEquals(0f, c.vx, 0f)
        c.run(1f, t, null)
        assertEquals(FollowMode.LOOK, c.mode)
    }

    @Test
    fun huggingStandsStill() {
        val c = FollowController(cling = true).apply { headSupported = true }
        val t = c.run(4f, 0, Sighting(0f, Legs.FEET, 0))
        c.reset()
        c.hugging = true
        c.run(3f, t, Sighting(0.5f, Legs.FEET, 0)) {
            assertEquals(0f, it.vx, 0f)
            assertEquals(0f, it.vtheta, 0f)
        }
        assertEquals(FollowMode.HUG, c.mode)
    }

    @Test
    fun coverMeansOneColourForAWhile() {
        val flat = IntArray(768) { 0xff7a6e60.toInt() }
        val (mean, std) = lumaStats(flat)
        assertEquals(0f, std, 0.01f)
        assertTrue(mean > 100f)
        val checks = IntArray(768) { if (it % 2 == 0) 0xff000000.toInt() else 0xffffffff.toInt() }
        assertEquals(127.5f, lumaStats(checks).second, 0.5f)

        val w = CoverWatch()
        var t = 0L
        repeat(10) { w.frame(120f, 40f, t); t += 60 }
        assertTrue(!w.covered(t))
        assertTrue("clear", w.clearFor(t) > 0)
        repeat(5) { w.frame(120f, 5f, t); t += 60 }
        assertTrue("under 0.4 s", !w.covered(t))
        repeat(3) { w.frame(15f, 30f, t); t += 60 }  // dark counts too
        assertTrue(w.covered(t))
        assertEquals(0L, w.clearFor(t))
        assertTrue("frames stop: not trusted", !w.covered(t + 2000))
        w.frame(120f, 40f, t)
        assertTrue(!w.covered(t))
        assertEquals("since the last covered frame", 560L, w.clearFor(t + 500))
    }

    @Test
    fun letsGoSoonerWhenTheyAreSeenWalkingOff() {
        assertTrue(!hugOver(1000, true))
        assertTrue(hugOver(RELEASE_SEEN_MS, true))
        assertTrue(!hugOver(5000, false))
        assertTrue(hugOver(RELEASE_ALONE_MS, false))
    }
}
