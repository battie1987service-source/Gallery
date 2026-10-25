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
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * Watches the front camera and reports when the user turns, tilts or nods their head, or opens or puckers their mouth,
 * depending on [mode].
 * A gesture fires once the movement passes the trigger value and re-arms only after the face returns to rest.
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

    // mouth opening as a fraction of the mouth width, and how much narrower the mouth gets when puckered
    private val mouthOpenTrigger = when (sensitivity) {
        HEAD_GESTURES_SENSITIVITY_LOW -> 0.45f
        HEAD_GESTURES_SENSITIVITY_HIGH -> 0.22f
        else -> 0.33f
    }
    private val mouthPuckerTrigger = when (sensitivity) {
        HEAD_GESTURES_SENSITIVITY_LOW -> 0.20f
        HEAD_GESTURES_SENSITIVITY_HIGH -> 0.09f
        else -> 0.14f
    }

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
                .apply {
                    if (mode == HEAD_GESTURES_MODE_MOUTH) {
                        setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                    }
                }
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
                    } else if (mode == HEAD_GESTURES_MODE_MOUTH) {
                        handleMouth(face)
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
        if (abs(angle) < neutralAngle) {
            isArmed = true
        } else if (abs(angle) > triggerAngle) {
            trigger((angle > 0) == forwardWhenPositive)
        }
    }

    // an open mouth ("a") goes to the next item, a puckered one ("u") to the previous
    private fun handleMouth(face: Face) {
        val upperLip = face.getContour(FaceContour.UPPER_LIP_TOP)?.points ?: return
        val upperLipInner = face.getContour(FaceContour.UPPER_LIP_BOTTOM)?.points ?: return
        val lowerLipInner = face.getContour(FaceContour.LOWER_LIP_TOP)?.points ?: return
        if (upperLip.size < 2 || upperLipInner.isEmpty() || lowerLipInner.isEmpty() || face.boundingBox.width() == 0) {
            return
        }

        val mouthWidth = abs(upperLip.last().x - upperLip.first().x)
        if (mouthWidth == 0f) {
            return
        }

        val opening = (lowerLipInner[lowerLipInner.size / 2].y - upperLipInner[upperLipInner.size / 2].y) / mouthWidth
        // the resting mouth width differs per person, so compare it with a slowly adapting baseline
        val widthRatio = mouthWidth / face.boundingBox.width()
        val restingWidthRatio = baseline ?: widthRatio
        val narrowing = 1 - widthRatio / restingWidthRatio

        val isAtRest = opening < mouthOpenTrigger * 0.4f && abs(narrowing) < mouthPuckerTrigger * 0.4f
        baseline = if (isAtRest) {
            restingWidthRatio + (widthRatio - restingWidthRatio) * BASELINE_SMOOTHING
        } else {
            restingWidthRatio
        }

        when {
            isAtRest -> isArmed = true
            opening > mouthOpenTrigger -> trigger(forward = true)
            narrowing > mouthPuckerTrigger && opening < mouthOpenTrigger * 0.6f -> trigger(forward = false)
        }
    }

    private fun trigger(forward: Boolean) {
        val now = SystemClock.elapsedRealtime()
        if (isRunning && isArmed && now - lastTriggerTime > COOLDOWN_MS) {
            isArmed = false
            lastTriggerTime = now
            callback(forward)
        }
    }
}
