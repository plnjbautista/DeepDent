import 'package:flutter/material.dart';
import 'dart:io';

class GingiCamScreen extends StatelessWidget {
  final String imagePath;
  const GingiCamScreen({super.key, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gingivitis Preview'),
        backgroundColor: colors.primary,
        foregroundColor: colors.onPrimary,
      ),
      body: Column(
        children: [
          Expanded(
            child: Image.file(
              File(imagePath),
              fit: BoxFit.contain,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: colors.primary,
                  foregroundColor: colors.onPrimary,
                ),
                onPressed: () {
                  Navigator.pop(context); // Retake
                },
                child: const Text('Retake'),
              ),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: colors.secondary,
                  foregroundColor: colors.onSecondary,
                ),
                onPressed: () {
                  // TODO: Proceed with this image
                },
                child: const Text('Use Photo'),
              ),
            ],
          ),
          const SizedBox(height: 50), // Optional: small space at the bottom
        ],
      ),
    );
  }
}

