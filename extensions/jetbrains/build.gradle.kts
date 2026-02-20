plugins {
    kotlin("jvm") version "1.9.24"
    id("org.jetbrains.intellij") version "1.17.4"
}

group = "com.deepseek.ide"
version = "0.0.1"

repositories {
    mavenCentral()
}

intellij {
    version.set("2024.1")
    type.set("IC")
}

tasks {
    patchPluginXml {
        sinceBuild.set("241")
        untilBuild.set("251.*")
    }

    runIde {
        jvmArgs = listOf("-Xmx2g")
    }
}

kotlin {
    jvmToolchain(17)
}
