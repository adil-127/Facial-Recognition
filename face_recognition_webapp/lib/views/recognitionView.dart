
// import 'dart:convert';
// import 'dart:typed_data';

// import 'package:flutter/material.dart';
// import 'package:flutter_webrtc/flutter_webrtc.dart';
// import 'package:http/http.dart' as http;

// class FRecognition extends StatefulWidget {
//   @override
//   State<StatefulWidget> createState() => _CaptureFrameSampleState();
// }

// class _CaptureFrameSampleState extends State<FRecognition> {
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


import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;

class FRecognition extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _CaptureFrameSampleState();
}

class _CaptureFrameSampleState extends State<FRecognition> {
  Uint8List? _data;
  var classification='';
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
    });

    // Convert frame to base64
    List frame=[];
     frame.add(base64.encode(_data!));
    print(frame);
    try {
      final body = jsonEncode(<String, dynamic>{
      'frame': frame,  
    });
      final response = await http.post(
        Uri.parse('http://127.0.0.1:8000/process-frame/'),
        headers: <String, String>{
          'Content-Type': 'application/json; charset=utf-8',
        },
        body: body
      );

      // Check if the request was successful
      if (response.statusCode == 200) {
        // Parse the JSON response
        final Map<String, dynamic> responseData = jsonDecode(response.body);
        final String status = responseData['status'];
        setState(() {
          
        classification = responseData['classification'];
        });

        print('Status: $status');
        print('Classification: $classification');
      } else {
        print('Failed to fetch data: ${response.statusCode}');
      }
    } catch (e) {
      print('Error during API call: $e');
    }
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

        backgroundColor: Color(0xff004d40),
        title: const Text('Capture to recognize'),
      ),
      floatingActionButton: FloatingActionButton(
                focusColor: Color(0xff80cbc4),
        backgroundColor: Color(0xff26a69a),
        onPressed: _captureFrame,
        child: Icon(Icons.camera_alt_outlined,color: Color(0xff004d40),),
      ),
      body:SingleChildScrollView(
        child: Column(
            children: [
              Center(
                child: _mediaStream == null
                    ? CircularProgressIndicator()
                    : Stack(
                      children: [
                          LayoutBuilder(
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
                          Positioned(
                            top: 10,
                            right: 50,
                            left: 50,
                            child: Text(classification,style: TextStyle(fontSize: 50,
                            color: classification=='UN_KNOWN' || classification=='Fake'? Colors.red:Color(0xff004d40)
                            ),))
                        ],
                    ),
              ),
           
            ],
        
           
           
        ),
      ),
    );
  }
}
