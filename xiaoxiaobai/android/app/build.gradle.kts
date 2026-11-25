plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.xiaoxiaobai.xxb"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.xiaoxiaobai.xxb"
        minSdk = 26
        targetSdk = 35
        versionCode = 4
        versionName = "1.0.3"
        // ONNX Runtime (follow mode) ships native code per ABI: phones only.
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Installable as is; sign with your own key before publishing to a store.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        buildConfig = true
    }
}

/** The bundled pages (robot list, pairing, Wi-Fi setup, Bluetooth remote) share the web UI's look and bridge. */
abstract class BundleWebui : DefaultTask() {
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val webui: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val out = outputDir.get().asFile.resolve("web")
        out.deleteRecursively()
        webui.asFileTree.visit {
            if (!isDirectory) file.copyTo(relativePath.getFile(out), overwrite = true)
        }
    }
}

val bundleWebui = tasks.register<BundleWebui>("bundleWebui") {
    webui.from(fileTree("../../webui") { include("css/app.css", "js/ui.js", "js/native.js", "js/components/joystick.js", "img/logo.svg") })
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(bundleWebui, BundleWebui::outputDir)
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.webkit:webkit:1.12.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
