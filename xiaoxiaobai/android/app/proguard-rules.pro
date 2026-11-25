# The web pages call these by name through window.XxbNative.
-keepclassmembers class com.xiaoxiaobai.xxb.Bridge {
    @android.webkit.JavascriptInterface <methods>;
}

# ONNX Runtime's native code looks its Java classes up by name.
-keep class ai.onnxruntime.** { *; }
