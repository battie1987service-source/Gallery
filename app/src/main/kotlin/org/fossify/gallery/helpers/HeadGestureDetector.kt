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
 * Watches the front camera and reports when the user turns, tilts or nods their head, depending on [mode].
 * A gesture fires once the angle passes the trigger angle and re-arms only after the head returns near the center.
 */
class HeadGestureDetector(
    private val activity: ComponentActivity,
    private val mode: Int,
    sensitivity: Int,
    private val callback: (forward: Boolean) -> Unit
) {
    companion object {
        private const val COOLDOWN_MS = 700L
        private const val BASELINE_SMOOTHING = 0.05f
    }

    // tilting or nodding the head has a smaller comfortable range than turning it
    private val triggerAngle = if (mode == HEAD_GESTURES_MODE_TILT || mode == HEAD_GESTURES_MODE_NOD) {
        when (sensitivity) {
            HEAD_GESTURES_SENSITIVITY_LOW -> 25f
            HEAD_GESTURES_SENSITIVITY_HIGH -> 10f
            else -> 17f
        }
    } else {
        when (sensitivity) {
            HEAD_GESTURES_SENSITIVITY_LOW -> 35f
            HEAD_GESTURES_SENSITIVITY_HIGH -> 15f
            else -> 25f
        }
    }
    private val neutralAngle = triggerAngle * 0.4f

    private var cameraProvider: ProcessCameraProvider? = null
    private var analysisExecutor: ExecutorService? = null
    private var faceDetector: FaceDetector? = null
    private var isArmed = true
    private var lastTriggerTime = 0L
    private var isRunning = false
    private var baseline: Float? = null

    fun start() {
        if (isRunning) {
            return
        }

        isRunning = true
        isArmed = false
        baseline = null
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
                    if (mode == HEAD_GESTURES_MODE_TILT) {
                        // the user tilting their head to their right rotates it counter-clockwise in the unmirrored frame
                        handleAngle(face.headEulerAngleZ, forwardWhenPositive = true)
                    } else if (mode == HEAD_GESTURES_MODE_NOD) {
                        // looking down gives a negative pitch, measured from the usual resting pitch since
                        // the phone is rarely held exactly at eye level
                        handleAngle(getPitchFromBaseline(face.headEulerAngleX), forwardWhenPositive = false)
                    } else {
                        // front camera frames are not mirrored, so the user turning to their right gives a negative yaw
                        handleAngle(face.headEulerAngleY, forwardWhenPositive = false)
                    }
                }
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    private fun getPitchFromBaseline(pitch: Float): Float {
        val currentBaseline = baseline ?: pitch
        val delta = pitch - currentBaseline
        // follow slow posture changes only while the head is at rest, so a nod does not shift the baseline
        baseline = if (abs(delta) < neutralAngle) {
            currentBaseline + delta * BASELINE_SMOOTHING
        } else {
            currentBaseline
        }
        return delta
    }

    // runs on the main thread, ML Kit delivers results there by default
    private fun handleAngle(angle: Float, forwardWhenPositive: Boolean) {
        if (!isRunning) {
            return
        }

        if (abs(angle) < neutralAngle) {
            isArmed = true
            return
        }

        val now = SystemClock.elapsedRealtime()
        if (isArmed && abs(angle) > triggerAngle && now - lastTriggerTime > COOLDOWN_MS) {
            isArmed = false
            lastTriggerTime = now
            callback((angle > 0) == forwardWhenPositive)
        }
    }
}
