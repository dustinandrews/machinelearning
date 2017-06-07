import vjoy

joyState = [vjoy.JoystickState(), vjoy.JoystickState()]

vjoy.Initialize()
joyState[0].XAxis = vjoy.AXIS_MAX
joyState[0].YAxis = vjoy.AXIS_MIN
joyState[0].ZAxis = 32767
joyState[0].Buttons = 0xAAAAAAAA
joyState[0].POV = (4 << 12) | (vjoy.POV_NIL << 8) | (4 << 4) | 4

vjoy.UpdateJoyState(0, joyState[0])
vjoy.Shutdown()
