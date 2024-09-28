<script lang="ts">
	import { onMount } from 'svelte';
	let finalLookSrc: string = '';
	let videoStream: MediaStream | null = null;
	let capturedImageSrc: string = '';
	let canvas: HTMLCanvasElement | null = null;
	let video: HTMLVideoElement | null = null;
	let aspectRatio: number = 4 / 3; // Default aspect ratio for webcam

	// Function to handle image upload for final look
	function handleImageChange(event: Event) {
		const target = event.target as HTMLInputElement;
		const file = target.files ? target.files[0] : null;

		if (file) {
			const reader = new FileReader();
			reader.onload = (e: ProgressEvent<FileReader>) => {
				if (e.target) {
					finalLookSrc = e.target.result as string;
				}
			};
			reader.readAsDataURL(file);
		}
	}

	// Start the webcam
	async function startCamera() {
		try {
			// Request the front camera by setting the facingMode constraint
			const constraints = {
				video: {
					facingMode: 'user' // This requests the front camera
				}
			};

			videoStream = await navigator.mediaDevices.getUserMedia(constraints);
			if (video) {
				video.srcObject = videoStream;
				video.onloadedmetadata = () => {
					aspectRatio = video.videoWidth / video.videoHeight; // Adjust aspect ratio to the camera's
					video.play();
					adjustCanvasSize();
				};
			}
		} catch (error) {
			console.error('Error accessing the camera: ', error);
		}
	}

	// Adjust canvas size to match video aspect ratio
	function adjustCanvasSize() {
		if (canvas && video) {
			canvas.width = video.videoWidth;
			canvas.height = video.videoHeight;
		}
	}

	// Draw lines on the canvas to indicate nose and eye direction
	function drawOverlay() {
		if (canvas && video) {
			const ctx = canvas.getContext('2d');
			if (ctx) {
				// Clear the canvas and draw the video frame
				ctx.clearRect(0, 0, canvas.width, canvas.height);
				ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

				// Draw eye direction line (green) spanning full width
				ctx.strokeStyle = 'green';
				ctx.lineWidth = 2;
				ctx.beginPath();
				ctx.moveTo(0, canvas.height * 0.3); // Start at the left edge
				ctx.lineTo(canvas.width, canvas.height * 0.3); // End at the right edge
				ctx.stroke();

				// Draw nose direction line (blue) spanning full height
				ctx.strokeStyle = 'blue';
				ctx.lineWidth = 2;
				ctx.beginPath();
				ctx.moveTo(canvas.width * 0.5, 0); // Start at the top edge
				ctx.lineTo(canvas.width * 0.5, canvas.height); // End at the bottom edge
				ctx.stroke();
			}
		}
	}

	// Capture the frame with the overlay
	function captureFrame() {
		if (canvas) {
			capturedImageSrc = canvas.toDataURL('image/png');
		}
	}

	// On mount, start the camera and set the drawing interval
	onMount(() => {
		startCamera();
		const interval = setInterval(drawOverlay, 100); // Draw the overlay every 100ms
		return () => clearInterval(interval); // Cleanup interval on unmount
	});
</script>

<div class="container">
	<!-- Final look image upload -->
	<input
		type="file"
		accept="image/*"
		on:change={handleImageChange}
		class="m-4 btn btn-lg variant-filled"
	/>

	<!-- Display uploaded final look image -->
	{#if finalLookSrc}
		<img class="video-preview" src={finalLookSrc} alt="Final Look Image" />
	{/if}

	<!-- Webcam video and overlay canvas -->
	<div class="video-wrapper" style="aspect-ratio: {aspectRatio}">
		<video bind:this={video} class="video-preview" autoplay muted playsinline></video>
		<canvas bind:this={canvas} class="canvas-preview"></canvas>

		<!-- Button to capture frame, floating at the bottom -->
		<button class="m-4 btn btn-lg variant-filled capture-button" on:click={captureFrame}
			>Capture Frame</button
		>
	</div>

	<!-- Display captured frame -->
	{#if capturedImageSrc}
		<img class="captured-image" src={capturedImageSrc} alt="Captured Frame" />
	{/if}
</div>

<style>
	.container {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 100vh;
		gap: 20px;
	}

	.video-preview,
	.canvas-preview,
	.captured-image {
		display: block;
		max-width: 100%;
	}

	.video-wrapper {
		position: relative;
		display: inline-block;
		width: 100%;
		max-width: 500px;
	}

	canvas {
		position: absolute;
		top: 0;
		left: 0;
	}

	/* Floating capture button */
	.capture-button {
		position: absolute;
		bottom: 10px;
		left: 50%;
	}
</style>
