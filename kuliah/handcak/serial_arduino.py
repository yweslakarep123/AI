import time
import serial


arduino = serial.Serial('COM4', 115200, timeout=1)

while True :
    arduino.write(bytes("1", 'utf-8')) 
    time.sleep(0.5)
    print("sudah jalan")
    time.sleep(2)
