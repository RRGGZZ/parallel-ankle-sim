import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco
import mujoco.viewer
import time
import math

# 替换成你自己的 .xml 模型路径
model = mujoco.MjModel.from_xml_path("adam_pro/adam_pro.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
# try:
# 	model.opt.gravity[:] = [0.0, 0.0, 0.0]
# except Exception:
# 	pass

# def actuator_id(name: str) -> int:
# 	aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
# 	if aid < 0:
# 		raise ValueError(f"actuator not found: {name}")
# 	return aid

# left_1 = actuator_id("ankleMotor_Left_1")
# left_2 = actuator_id("ankleMotor_Left_2")
# right_1 = actuator_id("ankleMotor_Right_1")
# right_2 = actuator_id("ankleMotor_Right_2")

# with mujoco.viewer.launch_passive(model, data) as viewer:
# 	t0 = time.time()
# 	while viewer.is_running():
# 		t = time.time() - t0
# 		u1 = 8.0 * math.sin(2.0 * math.pi * 0.35 * t)
# 		u2 = 8.0 * math.sin(2.0 * math.pi * 0.35 * t + math.pi)

# 		data.ctrl[left_1] = u1
# 		data.ctrl[left_2] = u2
# 		data.ctrl[right_1] = u1
# 		data.ctrl[right_2] = u2

# 		mujoco.mj_step(model, data)
# 		viewer.sync()
# 		time.sleep(0.001)