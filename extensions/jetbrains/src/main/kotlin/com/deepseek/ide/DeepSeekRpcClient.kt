package com.deepseek.ide

import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.util.concurrent.atomic.AtomicLong

class DeepSeekRpcClient private constructor() {
    private var process: Process? = null
    private var reader: BufferedReader? = null
    private var writer: BufferedWriter? = null
    private val requestId = AtomicLong(1)

    @Synchronized
    fun status(): String {
        ensureStarted()
        val id = requestId.getAndIncrement()
        val payload = "{\"jsonrpc\":\"2.0\",\"id\":$id,\"method\":\"status\",\"params\":{}}"
        writer?.write(payload)
        writer?.newLine()
        writer?.flush()
        return reader?.readLine() ?: "{\"error\":\"no_response\"}"
    }

    @Synchronized
    fun shutdown() {
        process?.destroy()
        process = null
        reader = null
        writer = null
    }

    @Synchronized
    private fun ensureStarted() {
        if (process != null) {
            return
        }
        val pb = ProcessBuilder("deepseek", "serve", "--transport", "stdio")
        pb.redirectErrorStream(true)
        process = pb.start()
        reader = BufferedReader(InputStreamReader(process!!.inputStream))
        writer = BufferedWriter(OutputStreamWriter(process!!.outputStream))

        val initId = requestId.getAndIncrement()
        val initialize = "{\"jsonrpc\":\"2.0\",\"id\":$initId,\"method\":\"initialize\",\"params\":{\"client\":\"jetbrains\"}}"
        writer?.write(initialize)
        writer?.newLine()
        writer?.flush()
        reader?.readLine()
    }

    companion object {
        val shared: DeepSeekRpcClient by lazy { DeepSeekRpcClient() }
    }
}
