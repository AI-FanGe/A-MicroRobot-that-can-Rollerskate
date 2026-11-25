package com.xiaoxiaobai.xxb

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.widget.FrameLayout
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.webkit.WebResourceErrorCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import org.json.JSONObject

/**
 * One WebView: the bundled setup pages (robot list, pairing, Wi-Fi, Bluetooth remote) or the
 * robot's own web console at http://<robot>:8000/, both with window.XxbNative.
 */
class MainActivity : ComponentActivity(), BleManager.Listener {
    lateinit var web: WebView
        private set
    private lateinit var bridge: Bridge
    private lateinit var topScrim: View
    private lateinit var bottomScrim: View
    private val app get() = application as XxbApp

    private var consoleHost: String? = null
    private var clearHistory = false
    private var fileCallback: ValueCallback<Array<Uri>>? = null
    private var afterBluetooth: (() -> Unit)? = null

    private val assets by lazy {
        WebViewAssetLoader.Builder().addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this)).build()
    }

    private val pickFiles = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        fileCallback?.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(result.resultCode, result.data))
        fileCallback = null
    }

    private val askPermissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        app.ble.refresh()
        if (app.ble.hasPermissions()) continueBluetooth() else afterBluetooth = null
    }

    private val askEnable = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        app.ble.refresh()
        if (app.ble.bluetoothOn()) afterBluetooth?.invoke()
        afterBluetooth = null
    }

    private var afterNotifications: (() -> Unit)? = null

    // Follow mode starts either way: without the permission its notification is only hidden.
    private val askNotifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        afterNotifications?.invoke()
        afterNotifications = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)

        val root = FrameLayout(this)
        web = WebView(this)
        topScrim = View(this)
        bottomScrim = View(this)
        root.addView(web, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        root.addView(topScrim, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, Gravity.TOP))
        root.addView(bottomScrim, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, Gravity.BOTTOM))
        setContentView(root)
        // The page lays out between the system bars (and above the keyboard); the strips behind
        // the bars take the colours of the page's top bar and bottom tab bar.
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            val bottom = maxOf(bars.bottom, insets.getInsets(WindowInsetsCompat.Type.ime()).bottom)
            (web.layoutParams as FrameLayout.LayoutParams).setMargins(bars.left, bars.top, bars.right, bottom)
            web.requestLayout()
            topScrim.layoutParams.height = bars.top
            bottomScrim.layoutParams.height = bars.bottom
            topScrim.requestLayout()
            bottomScrim.requestLayout()
            WindowInsetsCompat.CONSUMED
        }

        setupWebView()
        applyTheme(app.store.theme)
        app.ble.listener = this
        if (savedInstanceState == null || web.restoreState(savedInstanceState) == null) web.loadUrl("$SETUP#/boot")

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (web.canGoBack()) web.goBack() else moveTaskToBack(true)
            }
        })
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        web.saveState(outState)
    }

    override fun onDestroy() {
        if (app.ble.listener === this) app.ble.listener = null
        web.destroy()
        super.onDestroy()
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            mediaPlaybackRequiresUserGesture = false
            allowFileAccess = false
            allowContentAccess = false
            userAgentString = "$userAgentString XxbApp/${BuildConfig.VERSION_NAME}"
        }
        bridge = Bridge(this, app)
        web.addJavascriptInterface(bridge, "XxbNative")
        web.webViewClient = object : WebViewClientCompat() {
            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                assets.shouldInterceptRequest(request.url)

            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                if (ours(request.url)) return false
                openExternal(request.url)
                return true
            }

            override fun onPageStarted(view: WebView, url: String, favicon: android.graphics.Bitmap?) {
                bridge.allowed = ours(Uri.parse(url))
                app.ble.pageChanged()
            }

            override fun onPageFinished(view: WebView, url: String) {
                if (clearHistory) {
                    clearHistory = false
                    view.clearHistory()
                }
            }

            override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceErrorCompat) {
                // The robot's console didn't load (Wi-Fi changed, robot off): back to the robot page.
                if (request.isForMainFrame && request.url.scheme == "http") {
                    view.loadUrl("$SETUP#/offline?url=" + Uri.encode(request.url.toString()))
                }
            }
        }
        web.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(view: WebView, callback: ValueCallback<Array<Uri>>, params: FileChooserParams): Boolean {
                fileCallback?.onReceiveValue(null)
                fileCallback = callback
                return try {
                    pickFiles.launch(params.createIntent())
                    true
                } catch (e: ActivityNotFoundException) {
                    fileCallback = null
                    false
                }
            }
        }
        web.setDownloadListener { url, userAgent, contentDisposition, mimeType, _ ->
            val text = try {
                "正在下载 " + Downloads.fromUrl(this, url, userAgent, contentDisposition, mimeType)
            } catch (e: Exception) {
                "下载失败：${e.message}"
            }
            Toast.makeText(this, text, Toast.LENGTH_SHORT).show()
        }
    }

    private fun ours(url: Uri): Boolean =
        url.host == ASSET_HOST || (url.scheme == "http" && url.host != null && url.host == consoleHost)

    private fun openExternal(url: Uri) {
        try {
            startActivity(Intent(Intent.ACTION_VIEW, url))
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, "没有可以打开这个链接的应用", Toast.LENGTH_SHORT).show()
        }
    }

    fun openConsole(url: String) = runOnUiThread {
        val uri = Uri.parse(url)
        if (uri.scheme != "http" || uri.host.isNullOrEmpty()) return@runOnUiThread
        consoleHost = uri.host
        clearHistory = true
        web.loadUrl(url)
    }

    fun openSetup(route: String) = runOnUiThread {
        web.loadUrl(SETUP + if (route.startsWith("#")) route else "#/$route")
    }

    fun applyTheme(theme: String) {
        val light = theme == "light"
        // --bg (top bar) and --bg-2 (bottom tab bar) in webui/css/app.css
        topScrim.setBackgroundColor(Color.parseColor(if (light) "#F4F6FA" else "#0B0D12"))
        bottomScrim.setBackgroundColor(Color.parseColor(if (light) "#EEF1F6" else "#0F1218"))
        web.setBackgroundColor(Color.parseColor(if (light) "#F4F6FA" else "#0B0D12"))
        window.decorView.setBackgroundColor(Color.parseColor(if (light) "#F4F6FA" else "#0B0D12"))
        WindowCompat.getInsetsController(window, window.decorView).apply {
            isAppearanceLightStatusBars = light
            isAppearanceLightNavigationBars = light
        }
    }

    fun hasBluetooth() = getSystemService(BluetoothManager::class.java)?.adapter != null

    /** Run [action] once the Bluetooth permissions are granted and Bluetooth is on. */
    fun withBluetooth(action: () -> Unit) = runOnUiThread {
        if (!hasBluetooth()) {
            Toast.makeText(this, "这台手机没有蓝牙", Toast.LENGTH_SHORT).show()
            return@runOnUiThread
        }
        afterBluetooth = action
        if (!app.ble.hasPermissions()) askPermissions.launch(BleManager.PERMISSIONS) else continueBluetooth()
    }

    private fun continueBluetooth() {
        val action = afterBluetooth ?: return
        if (app.ble.bluetoothOn()) {
            afterBluetooth = null
            action()
            return
        }
        try {
            askEnable.launch(Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE))
        } catch (e: Exception) {
            afterBluetooth = null
            Toast.makeText(this, "请在系统设置里打开蓝牙", Toast.LENGTH_LONG).show()
        }
    }

    fun startFollow(robotUrl: String, fast: Float, slow: Float, cling: Boolean? = null) = runOnUiThread {
        val start = { FollowService.start(this, robotUrl, fast, slow, cling) }
        val granted = Build.VERSION.SDK_INT < 33 ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        if (granted) {
            start()
        } else {
            afterNotifications = start
            askNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    fun batteryExempt(): Boolean = getSystemService(PowerManager::class.java).isIgnoringBatteryOptimizations(packageName)

    @SuppressLint("BatteryLife")
    fun askBatteryExempt() {
        if (batteryExempt()) return
        try {
            startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName")))
        } catch (e: ActivityNotFoundException) {
            openAppSettings()
        }
    }

    fun openAppSettings() {
        try {
            startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName")))
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, "打不开系统设置", Toast.LENGTH_SHORT).show()
        }
    }

    /** window.__xxb.<fn>(args…) on the current page. */
    fun callJs(fn: String, vararg args: Any) {
        val list = args.joinToString(",") { if (it is Number) it.toString() else JSONObject.quote(it.toString()) }
        val script = "window.__xxb&&window.__xxb.$fn($list)"
        runOnUiThread { web.evaluateJavascript(script, null) }
    }

    override fun onState(json: String) = callJs("state", json)
    override fun onReply(jsId: Int, json: String) = callJs("resolve", jsId, json)
    override fun onEvent(json: String) = callJs("event", json)
    override fun onScan(json: String) = callJs("scan", json)

    companion object {
        private const val ASSET_HOST = "appassets.androidplatform.net"
        const val SETUP = "https://$ASSET_HOST/assets/app/setup.html"
    }
}
