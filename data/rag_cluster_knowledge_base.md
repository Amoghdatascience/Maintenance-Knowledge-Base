# RAG Cluster Knowledge Base

This file is a retrieval-oriented knowledge base for aviation maintenance clusters.
Evidence-backed entries are based on uploaded cluster summaries; other entries are inferred from the current cluster taxonomy and names.

## c_0 — inspection routine scheduled inspection
- Parent: inspection
- Status: evidence-backed
- Summary: Scheduled inspections, phase inspections, inspection time, AD/SB/ICA compliance, and records where the corrective action is inspection/compliance oriented rather than repair-oriented.
- Retrieval keywords: inspection due, insp time, phase insp, ad, service bulletin, ica, next due, no defects noted
- Include when:
  - Inspection due/completed
  - AD/SB/ICA compliance inspections
  - Inspector/admin time tied to inspection workflow
- Exclude when:
  - Repair records that only mention inspection incidentally
  - inspection panel/component issues better classified elsewhere
- Example patterns:
  - INSP TIME -> INSPECTED
  - PHASE 2 INSP DUE -> C/W INSP
  - AD ... FUEL INJ LINES INSP DUE -> INSP ... NO DEFECTS FOUND

## c_1 — aircraft start issue hard start no start starter no crank
- Parent: start_system
- Status: evidence-backed
- Summary: Start failures, hard starts, no-start/no-crank conditions, starter inoperative or not turning, and assisted-start situations.
- Retrieval keywords: would not start, could not start, hard start, starter won't turn, starter inop, assisted start, no crank
- Include when:
  - Aircraft would not start
  - Starter motor faults
  - Assisted starts due to start-system issues
- Exclude when:
  - External start cart/power cases if clearly external-start specific (c_41)
  - Battery-only issues if clearly electrical and not start-system behavior
- Example patterns:
  - REQUIRES ASSISTED START -> ASSISTED START OF ENGINE
  - STARTER WON'T TURN -> REMOVED STARTER & INSTALLED ...
  - A/C WOULD NOT START -> STARTED A/C

## c_2 — baffle bolt
- Parent: baffle
- Status: evidence-backed
- Summary: Baffle problems centered on bolts, missing mount bolts, bolt holes, or cracks adjacent to attach bolts.
- Retrieval keywords: baffle bolt, missing bolt, mount bolt, bolt hole, alternator attach bolt, oil cooler bolt
- Include when:
  - Missing baffle bolts
  - Baffle cracks at bolt locations
  - Baffle bolt-hole/rivet repairs tied to bolts
- Exclude when:
  - Pure screw cases (c_8)
  - pure bracket cases (c_3)
- Example patterns:
  - #2 CYL SIDE BAFFLE MISSING BOLT -> INSTALLED BOLT
  - BAFFLE CRACKED AT LOWER LEFT BOLT OF OIL COOLER -> STOP DRILLED BAFFLE

## c_3 — baffle bracket
- Parent: baffle
- Status: evidence-backed
- Summary: Baffle bracket defects including broken/sheared/pulled rivets on brackets, support brackets, and mounting bracket repairs.
- Retrieval keywords: baffle bracket, support bracket, mounting bracket, bracket rivets, bracket screw
- Include when:
  - Bracket rivet failures
  - Broken bracket hardware
  - Repairs focused on bracket rather than generic baffle skin
- Exclude when:
  - Generic cracks without bracket focus (c_4)
  - mount-specific screw/rivet cases (c_5)
- Example patterns:
  - AFT BAFFLE BRACKET RIVETS SHEARED -> REPLACED RIVETS
  - R/H BACK BAFFLE BRACKET IS MISSING A RIVET -> INSTALLED SCREWS ON BAFFLE BRACKET

## c_4 — baffle crack damage loose missing
- Parent: baffle
- Status: evidence-backed
- Summary: General baffle damage cluster for cracks, missing hardware, loose sections, and other non-specific baffle damage when not more specifically bolt/bracket/plug/screw/etc.
- Retrieval keywords: baffle cracked, stop drilled, missing hardware, loose baffle, damaged baffle
- Include when:
  - General cracked baffles
  - Missing/loose baffle hardware not clearly a bolt/screw/rivet child
  - Patch and stop-drill repairs
- Exclude when:
  - Specific plug/screw/rivet/bolt/bracket cases if explicit
- Example patterns:
  - #4 REAR ENGINE BAFFLE CRACKED -> STOP DRILLED CRACK
  - HARDWARE MISSING ON R/H AFT BAFFLE -> INSTALLED NEW HARDWARE

## c_5 — baffle mount
- Parent: baffle
- Status: evidence-backed
- Summary: Baffle mount issues such as mount screws, mount rivets, mount looseness, and cracks specifically by a mount point.
- Retrieval keywords: baffle mount, mount screw, mount rivets, mount loose, oil cooler mount
- Include when:
  - Rear/aft baffle mount loose
  - Mount screw/rivet sheared
  - Cracks by oil-cooler or baffle mount
- Exclude when:
  - General screws not clearly mount-related (c_8)
  - general bracket cases (c_3)
- Example patterns:
  - CYL #3 AFT BAFFLE MOUNT LOOSE -> FABRICATED PATCH
  - R/H REAR BAFFLE MOUNT SCREW IS MISSING -> INSTALLED NEW SCREW

## c_6 — baffle plug
- Parent: baffle
- Status: evidence-backed
- Summary: Baffle plug, spark-plug access plug, plug-hole, button-plug, and worn/broken/missing plug records.
- Retrieval keywords: baffle plug, spark plug access plug, button plug, plug tabs broken, plug hole enlarged
- Include when:
  - Worn/missing/broken baffle plugs
  - Plug-hole repairs
  - Spark-plug access plug replacements
- Exclude when:
  - Spark plugs for ignition system (c_39) when not baffle related
- Example patterns:
  - BAFFLE PLUG IS MISSING -> INSTALLED NEW PLUG
  - SPARK PLUG BAFFLE HOLE PLUGS ARE WORN -> REMOVED & REPLACED PLUGS

## c_7 — baffle rivet
- Parent: baffle
- Status: evidence-backed
- Summary: Baffle rivet failures including broken/sheared/pulled rivets, seal rivets, and rivet-based fastening repairs on baffles.
- Retrieval keywords: baffle rivet, sheared rivet, broken rivet, seal rivet, pulled through
- Include when:
  - Explicit rivet failures on baffles
  - Rivet-to-screw substitutions on baffles
  - Baffle seal rivet issues
- Exclude when:
  - Bracket rivet cases when bracket is the focus (c_3)
- Example patterns:
  - R/H REAR ENGINE BAFFLE RIVET BROKEN -> INSTALLED NEW RIVETS
  - BAFFLE SEAL RIVET IS BROKEN -> REPLACED W/ A SCREW

## c_8 — baffle screw
- Parent: baffle
- Status: evidence-backed
- Summary: Baffle screw problems such as missing/loose screws, mounting screw issues, and screw replacement/resecure actions.
- Retrieval keywords: baffle screw, missing screw, loose screw, mounting screw, tightened screw
- Include when:
  - Missing or loose baffle screws
  - Baffle screw replacement
  - Washer under screw head repairs
- Exclude when:
  - Mount-specific issues if the mount is the main fault (c_5)
  - bolt cases (c_2)
- Example patterns:
  - AFT BAFFLE SCREW LOOSE -> TIGHTENED BAFFLE SCREW
  - R/H BACK BAFFLE IS MISSING A SCREW -> INSTALLED NEW BAFFLE SCREW

## c_9 — baffle seal
- Parent: baffle
- Status: inferred-from-cluster-definition
- Summary: Baffle seal specific issues such as damaged, loose, missing, or worn seal material.

## c_10 — baffle spring
- Parent: baffle
- Status: inferred-from-cluster-definition
- Summary: Baffle spring issues including broken, weak, loose, or missing spring hardware.

## c_11 — baffle tie rod
- Parent: baffle
- Status: inferred-from-cluster-definition
- Summary: Baffle tie-rod related faults, looseness, breakage, or replacement.

## c_12 — cowling cowl damage loose
- Parent: cowling
- Status: inferred-from-cluster-definition
- Summary: Cowling/cowl damage or looseness, including cracked, loose, or missing cowl components.

## c_13 — cylinder compression low compression
- Parent: cylinder_Exhaust
- Status: inferred-from-cluster-definition
- Summary: Cylinder compression or low-compression findings.

## c_14 — cylinder crack failure
- Parent: cylinder_Exhaust
- Status: inferred-from-cluster-definition
- Summary: Cracked or failed cylinders.

## c_15 — exhaust valve stuck valve
- Parent: cylinder_Exhaust
- Status: inferred-from-cluster-definition
- Summary: Exhaust-valve faults or stuck-valve issues.

## c_16 — cylinder head temperature
- Parent: cylinder_Exhaust
- Status: inferred-from-cluster-definition
- Summary: Cylinder-head-temperature specific findings.

## c_17 — pushrod tube
- Parent: cylinder_Exhaust
- Status: inferred-from-cluster-definition
- Summary: Pushrod tube issues such as leaks, cracks, or replacement.

## c_18 — drain line tube
- Parent: engine_general
- Status: inferred-from-cluster-definition
- Summary: Drain-line or tube related engine faults.

## c_19 — carburetor carb
- Parent: engine_general
- Status: inferred-from-cluster-definition
- Summary: Carburetor/carb issues.

## c_20 — crankcase crankshaft firewall
- Parent: engine_general
- Status: inferred-from-cluster-definition
- Summary: Crankcase, crankshaft, or firewall related engine issues.

## c_21 — engine failure fire power loss engine quit
- Parent: engine_general
- Status: inferred-from-cluster-definition
- Summary: Serious engine failure, fire, quit, or power-loss events.

## c_22 — engine idle rpm issue
- Parent: engine_general
- Status: inferred-from-cluster-definition
- Summary: Idle or RPM control issues.

## c_23 — engine repair reinstall clean remove replace engine
- Parent: engine_general
- Status: inferred-from-cluster-definition
- Summary: Broad engine repair, remove/replace, reinstall, or cleaning actions.

## c_24 — engine run rough rough running misfire
- Parent: engine_general
- Status: inferred-from-cluster-definition
- Summary: Rough-running, stumble, or misfire engine issues.

## c_25 — engine seal tube bolt loose
- Parent: engine_general
- Status: inferred-from-cluster-definition
- Summary: Engine seal/tube/bolt looseness cluster.

## c_26 — propeller overspeed
- Parent: propeller
- Status: inferred-from-cluster-definition
- Summary: Propeller overspeed events.

## c_27 — induction leak induction system
- Parent: induction_intake
- Status: inferred-from-cluster-definition
- Summary: Induction leaks or general induction-system issues.

## c_28 — intake gasket
- Parent: induction_intake
- Status: inferred-from-cluster-definition
- Summary: Intake gasket faults or replacements.

## c_29 — intake tube boot seal
- Parent: induction_intake
- Status: inferred-from-cluster-definition
- Summary: Intake tube/boot/seal issues.

## c_30 — magneto ignition mag drop
- Parent: ignition
- Status: inferred-from-cluster-definition
- Summary: Magneto, ignition, or mag-drop issues.

## c_31 — mixture adjust mixture control
- Parent: fuel_control
- Status: inferred-from-cluster-definition
- Summary: Mixture control/adjustment cases.

## c_32 — oil cooler
- Parent: oil_system
- Status: inferred-from-cluster-definition
- Summary: Oil-cooler issues.

## c_33 — oil dipstick tube filler tube
- Parent: oil_system
- Status: inferred-from-cluster-definition
- Summary: Dipstick, filler tube, or oil-tube issues.

## c_34 — oil leak oil pressure oil temperature
- Parent: oil_system
- Status: inferred-from-cluster-definition
- Summary: Oil leak, oil pressure, or oil temperature issues.

## c_35 — oil return line
- Parent: oil_system
- Status: inferred-from-cluster-definition
- Summary: Oil return-line faults.

## c_36 — pilot reported in flight noticed
- Parent: pilot_reported
- Status: inferred-from-cluster-definition
- Summary: Pilot-reported in-flight discrepancy records.

## c_37 — rocker cover valve cover
- Parent: valve_cover
- Status: inferred-from-cluster-definition
- Summary: Rocker-cover or valve-cover issues.

## c_38 — sniffler valve
- Parent: valve_cover
- Status: inferred-from-cluster-definition
- Summary: Sniffler-valve related issues.

## c_39 — spark plug plug fouled
- Parent: ignition
- Status: inferred-from-cluster-definition
- Summary: Spark plug or fouled-plug issues.

## c_40 — battery issue low voltage weak battery charging issue
- Parent: electrical
- Status: inferred-from-cluster-definition
- Summary: Battery, low-voltage, weak battery, or charging-system issues.

## c_41 — external start issue external power ground power unit start
- Parent: start_system
- Status: inferred-from-cluster-definition
- Summary: External-start, external-power, or GPU-assisted start cases.

## c_42 — aileron
- Parent: flight_control
- Status: inferred-from-cluster-definition
- Summary: Aileron-related flight-control issue.

## c_43 — appearance cleaning paint wash dirty exterior clean fuselage surface finish cosmetic exterior surface
- Parent: appearance_cleaning
- Status: inferred-from-cluster-definition
- Summary: Cosmetic, paint, wash, clean, dirty-exterior, and surface-finish cases.

## c_44 — landing gear tire
- Parent: landing_gear_tire
- Status: inferred-from-cluster-definition
- Summary: Landing-gear or tire issue.

## c_45 — exhaust gas temperature
- Parent: cylinder_Exhaust
- Status: inferred-from-cluster-definition
- Summary: Exhaust-gas-temperature specific findings.

## c_46 — propeller damage
- Parent: propeller
- Status: inferred-from-cluster-definition
- Summary: Propeller damage.

## c_47 — elevator
- Parent: flight_control
- Status: inferred-from-cluster-definition
- Summary: Elevator-related flight-control issue.

## c_48 — rudder
- Parent: flight_control
- Status: inferred-from-cluster-definition
- Summary: Rudder-related flight-control issue.

## c_49 — flap
- Parent: flight_control
- Status: inferred-from-cluster-definition
- Summary: Flap-related flight-control issue.

## c_50 — spoiler
- Parent: flight_control
- Status: inferred-from-cluster-definition
- Summary: Spoiler-related flight-control issue.

## baffle_unspecified — baffle unspecified
- Parent: baffle
- Status: evidence-backed
- Summary: Baffle records that clearly belong to the baffle family but are not specific enough for bolt/bracket/crack/plug/rivet/screw/mount child routing, often involving generic cracked/worn fabrications and patches.
- Retrieval keywords: baffle cracked, fabricated patch, new baffle, grommet, plugs, muffler shroud, worn through
- Include when:
  - Generic baffle patch/fabrication work
  - Baffle grommet or general section repairs
  - Mixed baffle issues without a single dominant child token
- Exclude when:
  - Clear child-specific baffle part mentions
- Example patterns:
  - L/H BACK TOP BAFFLE IS CRACKED -> FABRICATED PATCH & INSTALLED
  - #3 AFT BAFFLE GROMMET TORN -> INSTALLED NEW GROMMET
