package org.fossify.gallery.helpers

import android.annotation.SuppressLint
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * Watches the front camera and reports when the user turns their head left or right.
 * A gesture fires once the yaw passes the trigger angle and re-arms only after the head returns near the center.
 */
class HeadGestureDetector(
    private val activity: ComponentActivity,
    sensitivity: Int,
    private val callback: (forward: Boolean) -> Unit
) {
    companion object {
        private const val COOLDOWN_MS = 700L
    }

    private val triggerAngle = when (sensitivity) {
        HEAD_GESTURES_SENSITIVITY_LOW -> 35f
        HEAD_GESTURES_SENSITIVITY_HIGH -> 15f
        else -> 25f
    }
    private val neutralAngle = triggerAngle * 0.4f

    private var cameraProvider: ProcessCameraProvider? = null
    private var analysisExecutor: ExecutorService? = null
    private var faceDetector: FaceDetector? = null
    private var isArmed = true
    private var lastTriggerTime = 0L
    private var isRunning = false

    fun start() {
        if (isRunning) {
            return
        }

        isRunning = true
        isArmed = false
        val providerFuture = ProcessCameraProvider.getInstance(activity)
        providerFuture.addListener({
            if (!isRunning || activity.isDestroyed) {
                return@addListener
            }

            val provider = try {
                providerFuture.get()
            } catch (e: Exception) {
                isRunning = false
                return@addListener
            }

            if (!provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
                isRunning = false
                return@addListener
            }

            val options = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setMinFaceSize(0.25f)
                .build()

            val detector = FaceDetection.getClient(options)
            val executor = Executors.newSingleThreadExecutor()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(executor) { imageProxy ->
                analyzeFrame(detector, imageProxy)
            }

            provider.unbindAll()
            provider.bindToLifecycle(activity, CameraSelector.DEFAULT_FRONT_CAMERA, analysis)
            cameraProvider = provider
            faceDetector = detector
            analysisExecutor = executor
        }, ContextCompat.getMainExecutor(activity))
    }

    fun stop() {
        isRunning = false
        cameraProvider?.unbindAll()
        cameraProvider = null
        faceDetector?.close()
        faceDetector = null
        analysisExecutor?.shutdown()
        analysisExecutor = null
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun analyzeFrame(detector: FaceDetector, imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage == null || !isRunning) {
            imageProxy.close()
            return
        }

        val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        detector.process(image)
            .addOnSuccessListener { faces ->
                val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                if (face != null) {
                    handleYaw(face.headEulerAngleY)
                }
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    // runs on the main thread, ML Kit delivers results there by default
    private fun handleYaw(yaw: Float) {
        if (!isRunning) {
            return
        }

        if (abs(yaw) < neutralAngle) {
            isArmed = true
            return
        }

        val now = SystemClock.elapsedRealtime()
        if (isArmed && abs(yaw) > triggerAngle && now - lastTriggerTime > COOLDOWN_MS) {
            isArmed = false
            lastTriggerTime = now
            // front camera frames are not mirrored, so the user turning to their right gives a negative yaw
            callback(yaw < 0)
        }
    }
}
