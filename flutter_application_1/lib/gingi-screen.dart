import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'gingi-cam.dart';

class GingiScreen extends StatefulWidget {
  const GingiScreen({super.key});

  @override
  State<GingiScreen> createState() => _GingiScreenState();
}

class _GingiScreenState extends State<GingiScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  FlashMode _flashMode = FlashMode.off;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final backCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
    );

    _controller = CameraController(backCamera, ResolutionPreset.medium);
    _initializeControllerFuture = _controller.initialize();
    await _initializeControllerFuture; // Wait for initialization
    await _controller.setFlashMode(FlashMode.off); // Explicitly turn flash off
    setState(() {});
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Image.asset('assets/logo/name.png', height: 150),
        centerTitle: true,
        backgroundColor: colors.onSurface,
        iconTheme: IconThemeData(color: colors.primary),
      ),
      body: FutureBuilder(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Stack(
              children: [
                // Fullscreen camera preview
                Positioned.fill(
                  child: CameraPreview(_controller),
                ),
                // 2:1 Ratio Frame Overlay, centered in the preview
                Center(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      double frameWidth = constraints.maxWidth * 0.8;
                      double frameHeight = frameWidth / 2;
                      return Container(
                        width: frameWidth,
                        height: frameHeight,
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: Colors.white.withOpacity(0.8),
                            width: 4,
                          ),
                          color: Colors.transparent,
                        ),
                      );
                    },
                  ),
                ),
                // Flash Toggle Button
                Positioned(
                  top: 40,
                  right: 30,
                  child: IconButton(
                    icon: Icon(
                      _flashMode == FlashMode.off
                          ? Icons.flash_off
                          : Icons.flash_on,
                      color: Colors.white,
                      size: 32,
                    ),
                    onPressed: () async {
                      setState(() {
                        _flashMode = _flashMode == FlashMode.off
                            ? FlashMode.torch
                            : FlashMode.off;
                      });
                      await _controller.setFlashMode(_flashMode);
                    },
                    tooltip: 'Toggle Flash',
                  ),
                ),
                // Take Picture Button (overlay, not pushing up the preview)
                Positioned(
                  bottom: 40,
                  left: 0,
                  right: 0,
                  child: Center(
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: colors.primary,
                        foregroundColor: colors.onPrimary,
                        shape: const CircleBorder(),
                        padding: const EdgeInsets.all(20),
                      ),
                      onPressed: () async {
                        try {
                          await _initializeControllerFuture;
                          final image = await _controller.takePicture();
                          if (!mounted) return;
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) =>
                                  GingiCamScreen(imagePath: image.path),
                            ),
                          );
                        } catch (e) {
                          // Handle error
                        }
                      },
                      child: const Icon(Icons.camera_alt, size: 32),
                    ),
                  ),
                ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
}
