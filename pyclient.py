import sys
import argparse
import socket
import keyboard
import time
import driver

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
parser.add_argument('--mode', action='store', dest='mode', default='ai',
                    help='Control mode (manual/ai) (default: ai)')

arguments = parser.parse_args()

# Print summary
print('TORCS AI Racing Client - Using Trained Model')
print('-------------------------------------------')
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('Control mode:', arguments.mode)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# One-second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0
verbose = False

# Initialize driver - force AI mode if not specified
ai_mode = arguments.mode.lower() == 'ai'  # Will be True for 'ai', False for anything else
d = driver.Driver(arguments.stage, ai_mode=True)  # Force AI mode regardless of argument

# Main client loop
while not shutdownClient:
    # Identification phase
    while True:
        print('Sending id to server:', arguments.id)
        buf = arguments.id + d.init()
        
        if verbose:
            print('Sending init string to server:', buf)

        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)

        try:
            buf, addr = sock.recvfrom(1000)
            buf_str = buf.decode()
            if verbose:
                print('Received data from server:', buf_str)
            if buf_str.find("***identified***") >= 0:
                print("Server identification successful")
                break
        except socket.error as msg:
            print("Didn't get response from server...")

    currentStep = 0
    
    # Racing phase
    while True:
        # Wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf_str = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...")
        
        if verbose:
            print('Received:', buf_str)
        
        # Check for shutdown command
        if buf_str != None and buf_str.find('***shutdown***') >= 0:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        
        # Check for restart command
        if buf_str != None and buf_str.find('***restart***') >= 0:
            d.onRestart()
            print('Client Restart')
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf_str != None:
                # Process the data through the AI driver
                try:
                    buf_str = d.drive(buf_str)
                except Exception as e:
                    print(f"Error in AI driving: {e}")
                    # Send neutral commands if driving fails
                    d.control.setAccel(0.0)
                    d.control.setBrake(0.0)
                    d.control.setSteer(0.0)
                    d.control.setGear(1)
                    buf_str = d.control.toMsg()
        else:
            buf_str = '(meta 1)'
        
        if verbose:
            print('Sending:', buf_str)
        
        if buf_str != None:
            try:
                sock.sendto(buf_str.encode(), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
    
    curEpisode += 1
    
    if curEpisode == arguments.max_episodes:
        shutdownClient = True

# Clean up
print("Closing connection and cleaning up...")
sock.close()
keyboard.unhook_all()
print("AI racing client shutdown complete.")