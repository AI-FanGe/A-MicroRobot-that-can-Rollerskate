pluginManagement {
    repositories {
        // dl.google.com is often unreachable from mainland China; the Aliyun mirrors carry the same artifacts.
        // The Aliyun google mirror answers 502 for artifacts it does not carry, which makes Gradle drop it.
        maven("https://maven.aliyun.com/repository/google") { content { includeGroupByRegex("androidx\\..*|com\\.android(\\..*)?|com\\.google\\.android\\..*|com\\.google\\.testing\\.platform|com\\.google\\.prefab") } }
        maven("https://maven.aliyun.com/repository/gradle-plugin")
        maven("https://maven.aliyun.com/repository/public")
        google()
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        maven("https://maven.aliyun.com/repository/google") { content { includeGroupByRegex("androidx\\..*|com\\.android(\\..*)?|com\\.google\\.android\\..*|com\\.google\\.testing\\.platform|com\\.google\\.prefab") } }
        maven("https://maven.aliyun.com/repository/public")
        google()
        mavenCentral()
    }
}

rootProject.name = "xiaoxiaobai"
include(":app")
