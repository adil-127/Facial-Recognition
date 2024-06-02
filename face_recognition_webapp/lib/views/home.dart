import 'package:face_recognition/utils/btn.dart';
import 'package:face_recognition/views/recognitionView.dart';
import 'package:face_recognition/views/register.dart';
import 'package:flutter/material.dart';
class home extends StatelessWidget {
  const home({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xff26a69a),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
           
          children: [
             InkWell(
              onTap: () {
                  Navigator.push(
                context, 
                MaterialPageRoute(builder: (context) => FRecognition()),
              );
                
              },
              child: customBtn(text: "Face Recognition")),
             SizedBox(
              height: 20,
             ),
             InkWell(
              onTap: () {
                Navigator.push(
                context, 
                MaterialPageRoute(builder: (context) => RegisterScreen()),
              );
              },
              child: customBtn(text: 'Register New User!',))
          ],
        ),
      ),
    );
  }
}