package com.xiaoxiaobai.xxb

import android.app.Application

/** Keeps the robot link alive across activity re-creation. */
class XxbApp : Application() {
    lateinit var store: RobotStore
        private set
    lateinit var ble: BleManager
        private set

    override fun onCreate() {
        super.onCreate()
        store = RobotStore(this)
        ble = BleManager(this, store)
    }
}
