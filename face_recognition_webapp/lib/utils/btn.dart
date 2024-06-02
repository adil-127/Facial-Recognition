import 'package:flutter/material.dart';
class customBtn extends StatelessWidget {
  final String text;

  const customBtn({Key? key, required this.text}) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return Container(
        width: 170,
        height: 50,
        decoration: BoxDecoration(  
          color: Color(0xff004d40),
          borderRadius: BorderRadius.circular(20)
        ),
        child: Center(child: Text(text,style: TextStyle(color: Color(0xffe0f2f1)),)),
      );
  }
}