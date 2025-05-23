import 'package:flutter/material.dart';
import 'gingi-ins.dart';
import 'perio-ins.dart';

class ChoiceScreen extends StatelessWidget {
  const ChoiceScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Image.asset(
          'assets/logo/name.png',
          height: 150,
        ),
        centerTitle: true,
        backgroundColor: colors.onSurface,
        iconTheme: IconThemeData(
          color: colors.primary, // Change this to your desired color
        ),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: colors.primary,
                foregroundColor: colors.onPrimary,
                textStyle: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'InterTight',
                ),
                shape: const RoundedRectangleBorder(
                  borderRadius: BorderRadius.zero,
                ),
                minimumSize: const Size(200, 50),
              ),
              onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const GingiInsScreen()),
                );  // TODO: Navigate to Gingivitis screen
              },
              child: const Text('Gingivitis'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: colors.secondary,
                foregroundColor: colors.onSecondary,
                textStyle: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'InterTight',
                ),
                shape: const RoundedRectangleBorder(
                  borderRadius: BorderRadius.zero,
                ),
                minimumSize: const Size(200, 50),
              ),
              onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const PerioInsScreen()),
                );  // TODO: Navigate to Gingivitis screen
              },
              child: const Text('Periodontitis'),
            ),
          ],
        ),
      ),
    );
  }
}