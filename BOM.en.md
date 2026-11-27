<p align="center">
  <a href="BOM.md">中文</a> | <a href="BOM.en.md"><strong>English</strong></a>
</p>

# Xiaoxiaobai Bill of Materials (BOM)

[Back to the project guide](README.en.md#assembly-guide)

This bill of materials lists the parts, models/specifications, and quantities needed to assemble one Xiaoxiaobai. Roller-skate parts are listed separately for builds that need skating functionality.

## Control and Head Electronics

| Part | Model / specification | Quantity | Purpose / notes |
| :--- | :--- | :---: | :--- |
| Main controller | Raspberry Pi Zero 2 W | 1 | Runs the control software, ONNX policies, and Web UI |
| microSD card | ≥16 GB | 1 | Stores the OS image and robot software |
| IMU module | YBX-BMI088 | 1 | Connects to the main controller over I2C; mount securely |
| Servo controller board | OpenRB | 1 | Connects to the main controller via USB |
| Bus servo | XL330; 1 Mbps bus rate | 20 | 12 for the legs, 6 for the arms, and 2 for the head; each servo comes with one 180 mm cable and screws |
| Head coprocessor | Seeed Studio XIAO ESP32S3 Sense | 1 | Includes a camera and microphone; see the [head firmware](xiaoxiaobai/firmware/xiao_face/xiao_face.ino) |
| Face display | 2-inch LCD, 320×240 | 1 | Displays facial animations |
| Audio amplifier | MAX98357 I2S amplifier module | 1 | Drives the head speaker |
| Speaker | Small speaker | 1 | Voice and music playback |

## Power and Charging

| Part | Model / specification | Quantity | Purpose / notes |
| :--- | :--- | :---: | :--- |
| Lithium-ion cell | Sony 18650, 3000 mAh per cell | 2 | Connected in series as a 2S1P battery pack |
| USB Type-C boost charging module | **2S, 8.4 V, 2 A** | 1 | USB Type-C input; select the **2S 2A** variant to charge two 18650 cells connected in series |
| DC-DC buck converter | **MP1584 adjustable buck module, with its output set and kept at 5 V** | 1 | Supplies 5 V to the main controller |
| Power switch | Three-pin slide switch | 1 | Main power control |

## Body Structure, Connectors, and Consumables

| Part | Model / specification | Quantity | Purpose / notes |
| :--- | :--- | :---: | :--- |
| Body bearing | **16 mm inner diameter × 22 mm outer diameter × 4 mm thickness** | **4** | Supports body shafts |
| Servo idler horn | 3D-printed part | 4 | Passive side of the dual-servo blocks at the hips and ankles |
| 3D-printed structural parts | PLA; default 0.2 mm layer height and 15% infill | 25 parts | See [`hardware/小小白3D打印.3mf`](hardware/小小白3D打印.3mf); per-part masses are in the [assembly guide](README.en.md#2-print-the-structural-parts) |
| Self-tapping screws | M2 / M2.5 | As needed | Structural assembly |
| Servo cables and splitters | Servo cables and one-to-two / one-to-three splitters | As needed | Servo daisy chains and branches to the limbs |
| USB, signal, and power cables | Match the connectors on the main controller, OpenRB, head board, and power modules | As needed | USB, I2C, UART, and power connections |
| Heat-shrink tubing / insulating materials | Size to match the wires and solder joints | As needed | Insulates connections |
| Thin rubber for the soles | Cut to fit the soles | Optional | Improves grip when walking |

## Roller-Skate Parts (Skating Functionality)

Bearing dimensions are listed as **inner diameter × outer diameter × thickness**, in mm.

| Part | Model / specification | Quantity | Purpose / notes |
| :--- | :--- | :---: | :--- |
| Roller-skate bearing | **5 × 10 × 4 mm** | **8** | Total for one robot's roller skates; counted separately from the 4 body bearings above |
| Roller skates | 3D-printed parts | 1 set | Used with the 8 roller-skate bearings listed above |

## Assembly Tools

A 3D printer, soldering iron, crimping tool, and other basic tools are needed. See the [assembly guide](README.en.md#assembly-guide) for wiring and servo configuration, and [software deployment](README.en.md#software-deployment) for software setup and persistent facial animations.
