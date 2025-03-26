#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2021 Bitcraze AB
#
#  AI-deck demo
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License along with
#  this program; if not, write to the Free Software Foundation, Inc., 51
#  Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#  Demo for showing streamed JPEG images from the AI-deck example.
#
#  By default this demo connects to the IP of the AI-deck example when in
#  Access point mode.
#
#  The demo works by opening a socket to the AI-deck, downloads a stream of
#  JPEG images and looks for start/end-of-frame for the streamed JPEG images.
#  Once an image has been fully downloaded it's rendered in the UI.
#
#  Note that the demo firmware is continously streaming JPEG files so a single
#  JPEG image is taken from the stream using the JPEG start-of-frame (0xFF 0xD8)
#  and the end-of-frame (0xFF 0xD9).

import argparse
import time
import socket,os,struct, time
import numpy as np
import cv2

def viewerFunction(output_queue1,output_queue2):


  deck_port = 5000
  deck_ip = "192.168.4.1"

  print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
  client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  client_socket.connect((deck_ip, deck_port))
  client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

  print("Socket connected")

  imgdata = None
  data_buffer = bytearray()

  def rx_bytes(size):
    data = bytearray()
    while len(data) < size:
      data.extend(client_socket.recv(size-len(data)))
    return data



  start = time.time()
  count = 0
  fps_avg_frame_count = 10

  #import hand_landmarker
  #hand_landmarks_module = hand_landmarker.Mediapipe_HandLandmarkerModule()
  try:
    while(1):
        # First get the info
        packetInfoRaw = rx_bytes(4)
        #print(packetInfoRaw)
        [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
        #print("Length is {}".format(length))
        #print("Route is 0x{:02X}->0x{:02X}".format(routing & 0xF, routing >> 4))
        #print("Function is 0x{:02X}".format(function))

        imgHeader = rx_bytes(length - 2)
        #print(imgHeader)
        #print("Length of data is {}".format(len(imgHeader)))
        [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

        if magic == 0xBC:
          #print("Magic is good")
          #print("Resolution is {}x{} with depth of {} byte(s)".format(width, height, depth))
          #print("Image format is {}".format(format))
          #print("Image size is {} bytes".format(size))

          # Now we start rx the image, this will be split up in packages of some size
          imgStream = bytearray()

          while len(imgStream) < size:
              packetInfoRaw = rx_bytes(4)
              [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
              #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
              chunk = rx_bytes(length - 2)
              imgStream.extend(chunk)
        
          count = count + 1
          # meanTimePerImage = (time.time()-start) / count
          # print("{}".format(meanTimePerImage))
          # print("{}".format(1/meanTimePerImage))
            
          # Calculate the FPS
          if count % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start)
            start = time.time()

            fps_text = 'FPS = {:.1f}'.format(fps)
            #print(fps_text)



          if format == 0:
              bayer_img = np.frombuffer(imgStream, dtype=np.uint8)   
              bayer_img.shape = (244, 324)
              color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2RGB)
              cv2.imshow('Raw', bayer_img)
              cv2.imshow('Color', color_img)

              if False:
                  cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
                  cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color_img)
              cv2.waitKey(1)
          else:
              with open("img.jpeg", "wb") as f:
                  f.write(imgStream)
              nparr = np.frombuffer(imgStream, np.uint8)
              decoded = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
              cv2.imshow('JPEG', decoded)
              cv2.waitKey(1)

          output_queue1.put(color_img)
          output_queue2.put(color_img)
          #print('queue size: ',output_queue1.qsize())

  
  finally:
     print('Process terminated')
     #client_socket.close()

     
