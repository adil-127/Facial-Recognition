// import 'dart:convert';
// import 'dart:typed_data';

// import 'package:flutter/material.dart';
// import 'package:flutter_webrtc/flutter_webrtc.dart';
// import 'package:http/http.dart' as http;

// class RegisterScreen extends StatefulWidget {
//   @override
//   State<StatefulWidget> createState() => _CaptureFrameSampleState();
// }

// class _CaptureFrameSampleState extends State<RegisterScreen> {
//   Uint8List? _data;

//   void _captureFrame() async {
//     final stream = await navigator.mediaDevices.getUserMedia({
//       'audio': false,
//       'video': true,
//     });

//     final track = stream.getVideoTracks().first;
//     final buffer = await track.captureFrame();
//     stream.getTracks().forEach((track) => track.stop());

//     setState(() {
//       _data = buffer.asUint8List();
//     });

//     // Convert frame to base64
//     List frame=[];
//      frame.add(base64.encode(_data!));
//     print(frame);
//     try {
//       final body = jsonEncode(<String, dynamic>{
//       'frame': frame,
//     });
//       final response = await http.post(
//         Uri.parse('http://127.0.0.1:8000/process-frame/'),
//         headers: <String, String>{
//           'Content-Type': 'application/json; charset=utf-8',
//         },
//         body: body
//       );

//       // Check if the request was successful
//       if (response.statusCode == 200) {
//         // Parse the JSON response
//         final Map<String, dynamic> responseData = jsonDecode(response.body);
//         final String status = responseData['status'];
//         final String classification = responseData['classification'];

//         print('Status: $status');
//         print('Classification: $classification');
//       } else {
//         print('Failed to fetch data: ${response.statusCode}');
//       }
//     } catch (e) {
//       print('Error during API call: $e');
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: const Text('Capture Frame'),
//       ),
//       floatingActionButton: FloatingActionButton(
//         onPressed: _captureFrame,
//         child: Icon(Icons.camera_alt_outlined),
//       ),
//       body: Builder(builder: (context) {
//         final data = _data;

//         if (data == null) {
//           return Container();
//         }
//         return Center(
//           child: Image.memory(
//             data,
//             fit: BoxFit.contain,
//             width: double.infinity,
//             height: double.infinity,
//           ),
//         );
//       }),
//     );
//   }
// }

// ignore: unused_import
import 'dart:convert';
import 'dart:typed_data';

import 'package:face_recognition/utils/btn.dart';
import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;


class RegisterScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<RegisterScreen> {
  Uint8List? _data;
  List frame = [];
  String? Name;
  String?status='';
  List encodeData=[];

  void _showNameDialog() {
    TextEditingController _nameController = TextEditingController();

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Enter your name'),
          content: TextField(
            controller: _nameController,
            decoration: InputDecoration(
              hintText: 'Name',
            ),
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text('Cancel'),
            ),
            TextButton(  
              onPressed: () {
                setState(() {
                  Name = _nameController.text;
                   sendData(Name!);
                  print(Name);
                });
                Navigator.of(context).pop();
              },
              child: Text('Submit'),
            ),
          ],
        );
      },
    );
  }


 void sendData(String name) async{
  // List encodeData=[];
  // for(int i=0;i<frame.length;i++){
  // encodeData.add(base64.encode(frame[i]));
  // }
  print("encoded Data                             :");
  print(encodeData);

  try {
        final body = jsonEncode(<String, dynamic>{
        'frame': encodeData,
        'name':name,  
                          });

        final response = await http.post(
          Uri.parse('http://127.0.0.1:8000/Register_User/'),
          headers: <String, String>{
            'Content-Type': 'application/json; charset=utf-8',
          },
          body: body
        );

        // Check if the request was successful
        if (response.statusCode == 200) {
          // Parse the JSON response
          final Map<String, dynamic> responseData = jsonDecode(response.body);
          setState(() {
            status = responseData['status'];
          });

          print('Status: $status');
        } else {
          print('Failed to fetch data: ${response.statusCode}');
        }
      } catch (e) {
        print('Error during API call: $e');
    }
 }

  void _captureFrame() async {
    final stream = await navigator.mediaDevices.getUserMedia({
      'audio': false,
      'video': true,
    });
    
    final track = stream.getVideoTracks().first;
    final buffer = await track.captureFrame();
    stream.getTracks().forEach((track) => track.stop());

    setState(() {
      _data = buffer.asUint8List();
      frame.add(_data);
           encodeData.add(base64.encode(_data!));
    print(encodeData);
    print(encodeData.length);
    });

    // Convert frame to base64

  }

   void removeItemAtIndex(int index) {
    setState(() {
      frame.removeAt(index);
      encodeData.removeAt(index);
    });
  }

  late RTCVideoRenderer _videoRenderer;
  MediaStream? _mediaStream;

  @override
  void initState() {
    super.initState();
    _initializeRenderer();
    _initializeCamera();
  }

  @override
  void dispose() {
    _videoRenderer.dispose();
    _mediaStream?.dispose();
    super.dispose();
  }

  Future<void> _initializeRenderer() async {
    _videoRenderer = RTCVideoRenderer();
    await _videoRenderer.initialize();
  }

  Future<void> _initializeCamera() async {
    final Map<String, dynamic> constraints = {
      'audio': false,
      'video': {
        'facingMode': 'user',
      },
    };

    try {
      _mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      _videoRenderer.srcObject = _mediaStream;
      setState(() {});

    } catch (e) {
      print('Error accessing camera: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Register User'),
        backgroundColor: Color(0xff004d40),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            Center(
              child: _mediaStream == null
                  ? CircularProgressIndicator()
                  : LayoutBuilder(
                      builder:
                          (BuildContext context, BoxConstraints constraints) {
                        final double aspectRatio =
                            _videoRenderer.videoWidth > 0 &&
                                    _videoRenderer.videoHeight > 0
                                ? _videoRenderer.videoWidth /
                                    _videoRenderer.videoHeight
                                : 1.0;
                                                          print('');
                                                          print('================================================');
                                                          print('');
                                                          

                        return AspectRatio(
                          aspectRatio: aspectRatio,
                          child: RTCVideoView(_videoRenderer),
                        );
                      },
                    ),
            ),
            SizedBox(
              height: 20,
            ),
            Container(
              height: 100,
              width: MediaQuery.of(context).size.width,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 5),
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: frame.length,
                  itemBuilder: (BuildContext context, int index) {
                    return Padding(
                      padding: const EdgeInsets.only(right: 20),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(
                            16.0), // Adjust the radius as needed
                        child: Stack(
                          children: [
                            Container(
                              child: AspectRatio(
                                aspectRatio: 16 / 9,
                                child: Image.memory(
                                  frame[index],
                                  fit: BoxFit.cover,
                                  width: double.infinity,
                                  height: double.infinity,
                                ),
                              ),
                            ),
                            Positioned(
                              top: 2,
                              right: 5,
                              child: InkWell(
                                onTap: () {
                                  removeItemAtIndex(index);
                                  print(frame.length);
                                },
                                child: Icon(Icons.cancel_rounded,color: Color(0xffb2dfdb),)))
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
            SizedBox(height: 20,),
            frame.length !=0?
            Center(
              child: InkWell(
                onTap: () {
                  _showNameDialog();
                  
                },
                child: customBtn(text: 'Upload the images')),
            ):Text(""),
          ],
        ),
     
     
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _captureFrame,
        focusColor: Color(0xff80cbc4),
        backgroundColor: Color(0xff26a69a),
        child: Icon(Icons.camera_alt_outlined,color: Color(0xff004d40)),
      ),
    );
  }
}
