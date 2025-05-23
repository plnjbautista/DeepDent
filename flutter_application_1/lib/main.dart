import 'package:flutter/material.dart';
import 'choice-screen.dart';

void main() {
  runApp(const MyApp());
}

// 🎨 Custom ColorScheme
final customColorScheme = const ColorScheme(
  brightness: Brightness.dark,
  primary: Color(0xFF7F2BB1),
  onPrimary: Color(0xFFD1D7E0),
  secondary: Color(0xFF554E6E),
  onSecondary: Color(0xFFD1D7E0),
  background: Color(0xFF2D283E),
  onBackground: Color(0xFFD1D7E0),
  surface: Color(0xFF4D4A5E),
  onSurface: Color(0xFFD1D7E0),
  error: Colors.red,
  onError: Colors.white,
);

// 🧩 Custom ThemeData
final ThemeData customTheme = ThemeData(
  colorScheme: customColorScheme,
  scaffoldBackgroundColor: customColorScheme.background,
  appBarTheme: AppBarTheme(
    backgroundColor: customColorScheme.primary,
    foregroundColor: customColorScheme.onPrimary,
  ),
  textTheme: const TextTheme(
    bodyMedium: TextStyle(
      color: Color(0xFFD1D7E0),
      fontSize: 16,
      fontFamily: 'InterTight', 
    ),
  ),
  useMaterial3: true,
);

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Custom ColorScheme Demo',
      theme: customTheme,
      home: const HomeScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Replace text with logo image
            Image.asset(
              'assets/logo/name-logo.png', // Update this path to your logo asset
              height: 150,
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: colors.primary,
                foregroundColor: colors.onPrimary,
                textStyle: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
                shape: const RoundedRectangleBorder(
                  borderRadius: BorderRadius.zero,
                ),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const ChoiceScreen()),
                );
              },
              child: const Text('Start'),
            ),
            // Removed the secondary button
          ],
        ),
      ),
    );
  }
}
