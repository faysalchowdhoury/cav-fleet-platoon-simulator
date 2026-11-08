In my simulations, 20% automated vehicles made a freeway lane 10% less efficient.

I built a mixed-autonomy simulator: human drivers on a Wiedemann 99 re-implementation (the Vissim freeway model, CC0-CC9), automated vehicles on the PATH ACC/CACC controllers with V2V packet loss and fallback, vectorised across the fleet. 280 runs, 0 collisions.

Queue-discharge capacity per lane:
🔹 0% CAV (W99 defaults): 2,634 veh/h
🔹 20-40% CAV: 2,382 (−10%)
🔹 60%: break-even
🔹 100% CACC: 4,339 (+65%)

Why the dip? CACC needs a connected leader. At low penetration, most automated vehicles sit behind a human and fall back to ACC with a 1.1 s gap, which is slower and less string-stable than W99 humans at CC1 = 0.9 s. The benefit arrives with platoons, not with automation.

Two more things I'd want any CAV study in Vissim to report:
1️⃣ V2V loss only matters once platoons exist. 50% packet loss was invisible at 20% penetration but cost 7% of capacity at 100%.
2️⃣ CC2 is as important as CC1. At CC1 = 0.9 s, moving CC2 from 2 to 6 m changed capacity from 3,036 to 2,348 veh/h and made disturbances grow more. Calibrating CC1 alone leaves a lot on the table.

Caveats: single lane, random CAV placement, and my own W99 implementation from published pseudocode (Vissim's is proprietary).

Repo: [link]

For those modeling CAVs in Vissim: what penetration-rate assumptions are agencies asking you to test?

#ConnectedVehicles #AutomatedVehicles #TrafficSimulation #VISSIM #TrafficEngineering #CACC #Python #MultiAgent
