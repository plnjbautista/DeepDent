import 'package:flutter/material.dart';
import 'gingi-screen.dart';


class GingiInsScreen extends StatelessWidget {
  const GingiInsScreen({super.key});

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
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(
              children: const [
                Icon(Icons.camera_alt, size: 40),
                SizedBox(width: 16),
                Expanded(
                  child: Text('Open your mouth to expose your teeth and gums.'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: Image.asset(
                    'assets/images/right.png',
                    height: 120,
                    fit: BoxFit.contain,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Image.asset(
                    'assets/images/wrong.png',
                    height: 120,
                    fit: BoxFit.contain,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: const [
                SizedBox(width: 16),
                Expanded(
                  child: Text(
                    '* Expose top and bottom part of the mouth. Gently pull your lips back to expose the gums. Ask assistance if needed.',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 12),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 32),
            Row(
              children: const [
                Icon(Icons.info_outline, size: 40),
                SizedBox(width: 16),
                Expanded(
                  child: Text('Ensure good lighting for accurate results.'),
                ),
              ],
            ),
            const SizedBox(height: 32),
            Row(
              children: const [
                Icon(Icons.check_circle_outline, size: 40),
                SizedBox(width: 16),
                Expanded(
                  child: Text('Follow on-screen instructions carefully.'),
                ),
              ],
            ),
            const SizedBox(height: 42),
            // Add Proceed button
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: colors.primary,
                foregroundColor: colors.onPrimary,
                textStyle: const TextStyle(
                  fontSize: 18,
                  fontFamily: 'InterTight',
                ),
                minimumSize: const Size(200, 50),
                shape: const RoundedRectangleBorder(
                  borderRadius: BorderRadius.zero,
                ),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const GingiScreen()),
                );
              },
              child: const Text('Proceed'),
            ),
          ],
        ),
      ),
    );
  }
}
