# Henry/hello_world
import omni.replicator.core as rep
from omni.replicator.core.scripts.utils import (
	ReplicatorItem,
	ReplicatorWrapper,
	create_node
)

@ReplicatorWrapper
def random_defect_color(
	numSamples: int = 1,
	target: tuple[3] = (43, .31, .93)
) -> ReplicatorItem:
	
	node = create_node("omni.graph.henry.OgnSampleDefectColor", target=target)
	
	return node

rep.settings.carb_settings("/omni/replicator/RTSubframes", 10)


with rep.new_layer():

	torus = rep.create.torus(semantics=[('class', 'torus')] , position=(0, -200 , 100))

	with rep.trigger.on_frame(max_execs=10):
		with torus:
			rep.modify.semantics([('class', 'defect')])
			rep.randomizer.color(
				colors = random_defect_color()
			)
