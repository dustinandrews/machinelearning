"""
Python bindings for Headsoft's VJoy driver version 1.2.
You can find VJoy here:  http://headsoft.com.au/index.php?category=vjoy
"""

from ctypes import *

__VERSION__ = 1.2
__AUTHOR__ = "Brent Taylor"
__CONTACT__ = "btaylor@fuzzylogicstudios.com"

POV_UP = 0
POV_RIGHT = 1
POV_DOWN = 2
POV_LEFT = 3
POV_NIL = 4

AXIS_MIN = -32767
AXIS_NIL = 0
AXIS_MAX = 32767

BUTTON_UP = 0
BUTTON_DOWN = 1

__vjoy = windll.VJoy

class JoystickState(Structure):
	"""
	Proposed Joystick State.

	A structure outlining the proposed state of the virtual joystick.
	All axis range from AXIS_MIN (default: -32767) to AXIS_MAX (default: 32767).

	"""
	_pack_ = 1
	_fields_ = [("ReportId", c_ubyte),
		("XAxis", c_short),
		("YAxis", c_short),
		("ZAxis", c_short),
		("XRotation", c_short),
		("YRotation", c_short),
		("ZRotation", c_short),
		("Slider", c_short),
		("Dial", c_short),
		("POV", c_ushort),
		("Buttons", c_uint32)]

def Initialize(name = "", serial = ""):
	"""
	Initialize the VJoy driver.

	Name and serial are provided for commercial usage.  Empty strings will place
	the driver in demo mode and will quit after five minutes of use.

	/param name Provided for commercial usage.  Leave blank for demo mode.
	/param serial Provided for commercial usage.  Leave blank for demo mode.
	/return Returns true for success and false for failure.

	"""
	return __vjoy.VJoy_Initialize(name, serial)

def Shutdown():
	"""
	Shutdown the VJoy driver.

	"""
	return __vjoy.VJoy_Shutdown()

def UpdateJoyState(Index, JoyState):
	"""
	Update the current joystick state specified by Index.
	
	Update the current joystick state specified by Index.  All axis range from AXIS_MIN
	(default: -32767) to AXIS_MAX (default: 32767).  Four POV's are provided each taking
	4 bits of a 32 bit integer.  POV values are POV_UP, POV_RIGHT, POV_DOWN, POV_LEFT and
	POV_NIL.

	/param Index Specifies which Joystick to update.  Two are provided as of 1.2, starting
	at Index of 0.
	/param JoyState A JoystickState structure with the proposed changes to the virtual
	Joystick.
	/return Returns true for success and false for failure.

	"""
	
	return __vjoy.VJoy_UpdateJoyState(Index, byref(JoyState))

def SetPOV(JoyState, Index, State):
	"""
	Helper function to set POV's by Index.

	/param JoyState A Joystick State structure where you want to make the POV changes.
	/param Index Which POV you wish to change.  Ranges between 0-3 for a total of 4 POV's.
	/param State The current state of the POV.  POV values are POV_UP, POV_RIGHT, POV_DOWN,
	POV_LEFT and POV_NIL.

	"""
	JoyState.POV &= ~(0xf << ((3 - Index) * 4))
	JoyState.POV |= State << ((3 - Index) * 4)

def SetButton(JoyState, Index, State):
	"""
	Helper function to set button state.

	/param JoyState A joystick State structure where you want to make the button changes.
	/param Index Which button you wish to change.  There are 32 buttons with a range of 0-31.
	/param State The current state of the button.  Values are BUTTON_UP or BUTTON_DOWN.

	"""
	if State == BUTTON_DOWN:
		JoyState.Buttons |= 1 << Index
	else:
		JoyState.Buttons &= ~(1 << Index)

