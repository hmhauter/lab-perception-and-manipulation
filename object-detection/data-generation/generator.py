import time
import omni.replicator.core as rep
import omni.usd
import carb

with rep.new_layer():
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
    carb.settings.get_settings().set("/app/asyncRendering", False)

    # Load in asset
    local_path = ""

    TABLE_USD = f"{local_path}/asset/Collected_EastRural_Table/EastRural_Table.usd"
    SPOON_SMALL_USD = f"{local_path}/asset/Collected_Spoon_Small/Spoon_Small.usd"
    SPOON_BIG_USD = f"{local_path}/asset/Collected_Spoon_Big/Spoon_Big.usd"
    FORK_SMALL_USD = f"{local_path}/asset/Collected_Fork_Small/Fork_Small.usd"
    FORK_BIG_USD = f"{local_path}/asset/Collected_Fork_Big/Fork_Big.usd"
    KNIFE_USD = f"{local_path}/asset/Collected_Knife/Knife.usd"
    SCENE_PATH = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/NVIDIA/Assets/Scenes/Templates/Outdoor/Puddles.usd"
    TEXTURE_LIST_SCENE = [
        "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr",
        "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/ZetoCG_com_WarehouseInterior2b.hdr",
        "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/bathroom_4k.hdr",
        "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/hospital_room_4k.hdr",
        "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/surgery_4k.hdr",
        "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/entrance_hall_4k.hdr",
        "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    ]
    PLATE_MATERIALS = [
        # "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Glass/Glass_Clear.mdl",
        # "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Glass/Glass_Colored.mdl",
        # "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Glass/Glass_Dirty.mdl",
        # "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Glass/Glass_Fritted.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Plastic_Standardized_Surface_Finish.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/PET_Clear.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Plastic_Thick_Translucent_Flakes.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Polycarbonate_Opaque.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Polyethylene_Opaque.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Polyethylene_Cloudy.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Polypropylene_Opaque.mdl",
        "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Polypropylene_Cloudy.mdl",
    ]
    PLATE_COLORS = [
        "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/props/riser_mPreviewSurface/diffuseColorTex.png",
        "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/props/riser_mPreviewSurface/opacityTex.png",
        "omniverse://localhost/NVIDIA/Usd_Explorer/Samples/Examples/2023_2/Factory/SubUSDs/textures/T_GlossyMetal_A1_ORM.png"

    ]

    # Insert here the paths to the plate USDC documents
    PLATE_1_USD = "Plate1.usdc"
    PLATE_2_USD = "Plate2.usdc"
    PLATE_3_USD = "Plate3.usdc"

    # Camera Parameters for 4 different cameras 
    # Camera 1 "from above"
    camera_1_conf = {
        "position": (46, 200, 45),
        "rotation": (-85, 0, 0),
        "focus_distance": 114,
        "focal_length": 27,
        "f_stop": 10,
        "resolution": (2048, 2048)
    }
    # Camera 2
    camera_2_conf = {
        "position": (46, 160, 230),
        "rotation": (-20, 0, 0),
        "focus_distance": 195,
        "focal_length": 27,
        "f_stop": 10,
        "resolution": (2048, 2048)
    }
    # Camera 3
    camera_3_conf = {
        "position": (150, 190, 200),
        "rotation": (-31, 30, 0),
        "focus_distance": 220,
        "focal_length": 27,
        "f_stop": 17,
        "resolution": (2048, 2048)
    }


    # Cultery path
    current_cultery = SPOON_SMALL_USD  # Change the item here e.g KNIFE_USD
    output_path = current_cultery.split(".")[0].split("/")[-1]

    def rect_lights(num=3):
        lights = rep.create.light(
            light_type="rect",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(1000, 5000),
            position=rep.distribution.uniform((0, 80, 0), (90, 200, 42)), #(45, 110, 0),
            rotation=rep.distribution.uniform((-100, -10, -10), (-80, 10, 10)), #(-90, 0, 0),
            scale=rep.distribution.uniform(50, 100),
            count=num,
            # texture=rep.distribution.choice(TEXTURE_LIST_SCENE),
        )
        return lights.node

    def dome_lights(num=1):
        lights = rep.create.light(
            light_type="dome",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(100, 1000),
            count=num,
            texture=rep.distribution.choice(TEXTURE_LIST_SCENE),
        )
        return lights.node

    def table():
        table = rep.create.from_usd(TABLE_USD)

        with table:
            rep.randomizer.materials(
                materials=[
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Wood/OSB_Wood_Splattered.mdl",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Wood/OSB_Wood.mdl",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Paper/Paper_Plain.mdl",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Paper/Cardboard.mdl",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Plastic/Styrofoam.mdl"
                ]
            )
            rep.modify.pose(
                position=(46, -0.0, 20),
                rotation=(0, -90, -90),
            )
        return table

    # Define randomizer function for CULTERY assets. This randomization includes placement and rotation of the assets on the surface.
    def random_distractors(size=15):
        instances = rep.randomizer.instantiate(rep.utils.get_usd_files(
            "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Kitchen/Kitchenware/", recursive=True), 
            size=1, mode='point_instance')

        with instances:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (0, 76.3651, 0), (90, 76.3651, 42)),
                rotation=rep.distribution.uniform(
                    (0, -90, -90), (0, -90, -90)),
            )
        return instances.node
    
    def plate_1(size=1):
        instances = rep.create.from_usd(usd=PLATE_1_USD,
            semantics=[('class', 'plate')]
        )

        with instances:
            rep.randomizer.materials(
                materials=PLATE_MATERIALS
            )
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (0, 76.3651, 0), (90, 76.3651, 42)),
                rotation=rep.distribution.uniform(
                    (90, 0, 0), (90, 0, 0)),
                scale=rep.distribution.uniform(0.012, 0.018),
            )
        return instances
    
    def plate_2(size=1):
        instances = rep.randomizer.instantiate(rep.utils.get_usd_files(PLATE_2_USD),
            size=rep.distribution.uniform(1, 2), mode='point_instance')

        with instances:
            rep.randomizer.materials(
                materials=PLATE_MATERIALS
            )
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (0, 76.3651, 0), (90, 76.3651, 42)),
                rotation=rep.distribution.uniform(
                    (90, 0, 0), (90, 0, 0)),
            )
        return instances.node
    
    def plate_3(size=1):
        instances = rep.create.from_usd(usd=PLATE_3_USD,
            semantics=[('class', 'plate')], count=2
        )

        with instances:
            rep.randomizer.materials(
                materials=PLATE_MATERIALS
            )
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (0, 76.3651, 0), (90, 76.3651, 42)),
                rotation=rep.distribution.uniform(
                    (0, -180, 0), (0, 180, 0)),
                scale=rep.distribution.uniform(0.12, 0.17),
            )
        return instances

    # scene = rep.create.from_usd(SCENE_PATH)

    # Register randomization
    rep.randomizer.register(table)
    rep.randomizer.register(random_distractors)
    rep.randomizer.register(rect_lights)
    rep.randomizer.register(dome_lights)
    rep.randomizer.register(plate_1)
    rep.randomizer.register(plate_2)
    rep.randomizer.register(plate_3)

    # Multiple setup cameras and attach it to render products
    camera_1 = rep.create.camera(focus_distance=camera_1_conf["focus_distance"], focal_length=camera_1_conf["focal_length"],
                               position=camera_1_conf["position"], rotation=camera_1_conf["rotation"], f_stop=camera_1_conf["f_stop"])
    camera_2 = rep.create.camera(focus_distance=camera_2_conf["focus_distance"], focal_length=camera_2_conf["focal_length"],
                               position=camera_2_conf["position"], rotation=camera_2_conf["rotation"], f_stop=camera_2_conf["f_stop"])
    camera_3 = rep.create.camera(focus_distance=camera_3_conf["focus_distance"], focal_length=camera_3_conf["focal_length"],
                               position=camera_3_conf["position"], rotation=camera_3_conf["rotation"], f_stop=camera_3_conf["f_stop"])

    render_product_1 = rep.create.render_product(camera_1, camera_1_conf["resolution"])
    render_product_2 = rep.create.render_product(camera_2, camera_2_conf["resolution"])
    render_product_3 = rep.create.render_product(camera_3, camera_3_conf["resolution"])




    with rep.trigger.on_frame(interval=5, num_frames=250, rt_subframes=10):
        rep.randomizer.table()
        rep.randomizer.rect_lights(3)
        rep.randomizer.dome_lights(1)
        rep.randomizer.random_distractors(1)
        # Decide which plate to render here (only plate 3 was used for the thesis)
        # rep.randomizer.plate_1()
        # rep.randomizer.plate_2()
        rep.randomizer.plate_3()


    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=f"./dataset_segm",
                      rgb=True, bounding_box_2d_tight=True, semantic_segmentation=True, distance_to_camera=True, distance_to_image_plane=True, normals=True, pointcloud=True, instance_id_segmentation=True )
    writer.attach([render_product_1, render_product_2, render_product_3])

    # Run the simulation graph
    rep.orchestrator.run()
