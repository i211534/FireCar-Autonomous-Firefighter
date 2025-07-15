import serial
import time

# Open the serial port to communicate with HC-05
# Replace '/dev/serial0' with your actual serial port if it's different
ser = serial.Serial('/dev/serial0', 9600)  # 9600 is default baud rate for HC-05

# Wait for the connection to establish
time.sleep(2)

# Data to send to the Bluetooth device
data = "f"
ser.write(data.encode())
time.sleep(2)

# data = "r"
# ser.write(data.encode())
# time.sleep(2)

# data = "l"
# ser.write(data.encode())
# time.sleep(2)



#data = "b"
#ser.write(data.encode())
#time.sleep(2)

data = "s"
ser.write(data.encode())
time.sleep(2)

# data = "a"
# ser.write(data.encode())
# time.sleep(2)


print("Data sent:", data)

# Close the serial port
ser.close()

