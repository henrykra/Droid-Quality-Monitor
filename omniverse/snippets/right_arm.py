# Henry/right_arm
import omni.replicator.core as rep
import numpy as np

DROID = "omniverse://localhost/Library/test.usd"
SCENE = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/LookDev/Decor_Wood.usd"

rep.settings.carb_settings("/omni/replicator/RTSubframes", 40)

from omni.replicator.core.scripts.utils import (
	ReplicatorItem,
	ReplicatorWrapper,
	create_node
)

@ReplicatorWrapper
def random_defect_color(
	numSamples: int = 1,
	target: tuple[3] = (.11, .6, .45)
) -> ReplicatorItem:
	
	node = create_node("omni.graph.henry.OgnSampleDefectColor", target=target)
	
	return node

with rep.new_layer():
	
	# creating static scene
	light = rep.create.light(rotation=(315, 0, 0), intensity=500, light_type="distant")
	
	camera = rep.create.camera(focal_length=75, position=(30, 12, 0), look_at=(0, 2, 0))
	
	scene = rep.create.from_usd(SCENE)
	
	droid = rep.create.from_usd(DROID, semantics=[('class', 'droid')])
	
	# attatch camera to render product
	render_product = rep.create.render_product(camera, resolution=(1024, 1024))
	
	# initialize and attach writer
	writer = rep.WriterRegistry.get("BasicWriter")
	writer.initialize(output_dir="right_arm_output", rgb=True, bounding_box_2d_tight=True)
	writer.attach([render_product])
	
	# define randomizing function
	def right_arm():
		right_arm = rep.get.prims(path_pattern='/Replicator/Ref_Xform_01/Ref/droid/droid/right_arm')
		
		with right_arm:
			rep.modify.semantics([('class', 'right_arm_defect')])
			rep.randomizer.color(colors=random_defect_color())
		
		return right_arm


	rep.randomizer.register(right_arm)

	
	with rep.trigger.on_frame(max_execs=100):
		with droid:
			rep.modify.pose(rotation=rep.distribution.normal((0, 90, 0), (0, 10, 0)))
		rep.randomizer.right_arm()

		
		
		