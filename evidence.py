"""Real evidence behind every TARP threshold: values recorded before or at real failures,
published TARPs in use, and the sources. No synthetic data."""
import pandas as pd

# ---------------------------------------------------------------- FOS of real dumps with a known outcome
FOS_CASES = pd.DataFrame([
    ["WCL internal dump, Wardha Valley (75 m at 43 deg)", "Failed", 0.80, "FEM strength reduction, lab peak strength", "Kainthola et al. 2011"],
    ["WCL internal dump, Wardha Valley (75 m at 43 deg)", "Failed", 1.07, "Bishop (this work), lab peak strength", "computed from Kainthola et al. 2011"],
    ["WCL internal dump, Wardha Valley (75 m at 43 deg)", "Failed", 0.72, "Bishop (this work), residual strength", "computed from Kainthola et al. 2011"],
    ["Jayant OCP dragline dump, NCL (85 m), 2008", "Failed", 0.94, "LEM with water table and weak interface", "Sharma & Roy 2015"],
    ["Coal waste dump, East Kalimantan", "Failed", 0.98, "Back-analysis at failure, after 353.5 mm of rain", "Dwinagara et al. 2024"],
    ["Coal waste dump, East Kalimantan", "Failed later", 1.45, "Same dump before the rain", "Dwinagara et al. 2024"],
    ["Singareni 'Mine A' dump No. 1, SCCL (90 m)", "Failed later", 1.49, "Design FOS (soft foundation not modelled)", "Poulsen et al. 2014"],
    ["Gevra OCP, SECL", "Standing", 1.94, "RS2, scientific study", "NIT Rourkela 2025"],
    ["Kusmunda OCP, SECL", "Standing", 1.616, "RS2, scientific study", "NIT Rourkela 2025"],
    ["Dipka OCP, SECL", "Standing", 1.701, "RS2, scientific study", "NIT Rourkela 2025"],
    ["Manikpur OCP, SECL", "Standing", 1.766, "RS2, scientific study", "NIT Rourkela 2025"],
    ["Chhal OCP, SECL", "Standing", 1.926, "RS2, scientific study", "NIT Rourkela 2025"],
    ["Baroud OCP, SECL", "Standing", 2.034, "RS2, scientific study", "NIT Rourkela 2025"],
    ["Shada OCM, SECL", "Standing", 1.888, "RS2, scientific study", "NIT Rourkela 2025"],
    ["Kanchan OCP, SECL", "Standing", 1.799, "RS2, scientific study", "NIT Rourkela 2025"],
], columns=["Dump", "Outcome", "FOS", "How the FOS was found", "Source"])

# ---------------------------------------------------------------- movement recorded before failures / in working TARPs
VEL_CASES = pd.DataFrame([
    ["Juna Kunada OC, WCL, 2017 (OB benches)", "Recorded before failure", 16, "Cracks opened from 1-2 mm to 50 mm in about 3 days; failed on day 4", "DGMS alert 2017"],
    ["Open-pit copper mine wall (radar)", "Recorded before failure", 60, "Alarm at 10 mm in 4 h (about 60 mm/day) sounded about 12 h before collapse", "Dick et al. 2015"],
    ["Jordan slide, Bingham Canyon", "Recorded before failure", 13, "About 13 mm/day for 4 months, then accelerated and failed (8 Oct 2019)", "Bakken et al. 2020"],
    ["O-Slide, Bingham Canyon", "Recorded (no failure)", 50, "Reached about 50 mm/day while accelerating; stopped after remediation", "Bakken et al. 2020"],
    ["Nine open-pit slope failures (copper, iron, coal mines)", "Recorded before failure", 60, "At least 60 mm/day 48 h before collapse in every case; 120-458 mm/day 24 h before", "Ryan & Call 1992"],
    ["Zavodni & Broadbent rule", "Threshold from failures", 50, "Above about 50 mm/day a slope is in the progressive stage; at least 48 h passed between 50 mm/day and collapse in all cases studied", "Ryan & Call 1992"],
    ["Slope failure 1 (copper mine, 210 m)", "Recorded before failure", 226, "Stage change at 226 mm/day about 36 h before collapse; 742 mm/day 8 h before", "Ryan & Call 1992"],
    ["Mine waste dumps, British Columbia", "Recorded before failure", 1000, "Velocity may exceed 1 m/day before failure; crest moves over 10 m", "Golder & Hungr 1995"],
    ["Bingham Canyon, pit-wide TARP", "Threshold in use", 20, "Velocity triggers 2-20 mm/day; displacement 10-50 mm", "Bakken et al. 2020"],
    ["Mount Milligan TARP (radar)", "Threshold in use", 72, "6 / 24 / 72 mm/day; above 72 mm/day stop and withdraw; radar warned about 24 h before a 2022 failure", "Magoon et al. 2024"],
    ["Coal spoils, Fording River (GPS)", "Threshold in use", 300, "Warning 150, alarm 300 mm/day (end-dumped mountain spoils)", "WSP 2023 for Teck"],
    ["NIT Rourkela scientific-study notes", "Threshold in use", 50, "Normal < 2, Watch 2-5, Caution 5-10, Alert 10-50, Alarm 50-100, Stop work > 100 mm/day", "NIT Rourkela notes"],
], columns=["Case", "Type", "Velocity (mm/day)", "What happened", "Source"])


# ---------------------------------------------------------------- nine real progressive slope failures (Ryan & Call 1992)
# velocity at the onset of the progressive stage, 48 h and 24 h before collapse (cm/day in the paper -> mm/day)
RYAN_CALL = pd.DataFrame({
    "Failure": [f"Slope failure {i}" for i in range(1, 10)],
    "Onset (mm/day)": [23, 16, 12, 9, 7, 3, 9, 15, 14],
    "48 h before (mm/day)": [201, 62, 102, 158, 60, 120, 164, 212, 140],
    "24 h before (mm/day)": [384, 179, 201, 330, 120, 182, 283, 458, 260],
})


def iv_backcheck(rc=RYAN_CALL):
    """Applies this TARP's rules to the nine real failures, using only what was known 24 h before
    collapse: velocity band (T3), velocity ratio (T4 Orange) and the inverse-velocity forecast
    (T4 Red) fitted through the 48 h and 24 h velocities."""
    out = []
    for r in rc.itertuples(index=False):
        v48, v24 = r[2], r[3]
        slope = (1 / v24 - 1 / v48) / 1.0
        t_fail = -1 - (1 / v24) / slope              # days relative to the real collapse
        out.append(dict(Failure=r[0], v48=v48, v24=v24, ratio=round(v24 / v48, 2),
                        forecast_h=round((t_fail + 1) * 24, 1), error_h=round(t_fail * 24, 1)))
    return pd.DataFrame(out)

ACC_CASES = pd.DataFrame([
    ["Short / long moving average crossover = onset of acceleration; fit time to failure only after it", "Carla et al. 2017", "T4 Yellow"],
    ["78 slope failures in Australian open-cut coal mines were each preceded by a short phase of acceleration lasting a few days", "Carla, Intrieri, Farina & Casagli 2017", "Why acceleration is a trigger"],
    ["In nine real failures velocity doubled in the last day (ratio of 24 h to 48 h velocity 1.5-2.9)", "Ryan & Call 1992", "T4 Orange at ratio 1.3 is reached in all nine"],
    ["Inverse velocity gave accurate forecasts 35 h before a one-million-tonne failure (25 Aug 2008); the radar alarm gave 15 h", "Dick et al. 2013", "T4 Red"],
    ["Inverse-velocity forecasts can be wrong by a few hours to several days", "Intrieri et al. 2019; Carla et al. 2017", "Red is set at 2 days, and velocity is checked alongside"],
    ["Short average over 3 readings, long over 6, on slope-radar data of three SECL mines", "Masood et al. 2025", "Averaging windows"],
    ["Velocity ratio 1.3-1.5 is a TARP trigger (isolate the area below)", "Bakken et al. 2020 (Bingham Canyon)", "T4 Orange at 1.3"],
    ["Inverse velocity predicting failure within 1-3 days (Jordan slide 36 h, Lark Quartzite 2 days) = remove people", "Bakken et al. 2020", "T4 Red at 2 days"],
    ["At the Jordan slide the real failure always came before the predicted time in the last 22 h", "Bakken et al. 2020", "Red is set early (2 days), not at the predicted hour"],
    ["Radar TARPs aim to alarm 12-24 h before collapse", "Shellam & Coggan 2020", "Red must leave hours for evacuation"],
    ["Rocks seen sliding about 3.5 h before the Rajmahal collapse", "Mongabay 2017", "T5 Red sign"],
], columns=["Finding", "Source", "Used for"])

RAIN_CASES = pd.DataFrame([
    ["Basundhara area (MCL), Aug 2013", "Heavy rain the day before; coal dump collapsed, 14 dead", "India Environment Portal 2013"],
    ["Bharatpur OCP (MCL), Jul 2019", "Monsoon; water held behind a rib and backfill; 4 dead. Most dumps and high walls in the region failed in the rainy season", "DGMS alert 2019"],
    ["Kusmunda OC (SECL), Jul 2020", "Earth slid during continuous rain while pumping", "Deccan Herald"],
    ["Wardha Valley mine (WCL)", "Rainwater raised pore pressure in a slope loaded by an OB dump", "Jhanwar 2010"],
    ["Coal waste dump, East Kalimantan", "353.5 mm in 17 days turned pore pressure positive; FOS 1.45 to 0.98 and failure", "Dwinagara et al. 2024"],
    ["Bingham Canyon TARP", "25 mm in 24 h raises the TARP; no mining below an instability in heavy rain", "Bakken et al. 2020"],
    ["IMD classes", "Heavy 64.5-115.5, very heavy 115.6-204.4, extremely heavy >= 204.5 mm in 24 h", "India Meteorological Department"],
], columns=["Case", "What happened", "Source"])

SIGN_CASES = pd.DataFrame([
    ["New tension cracks on the crest or bench", "Rajmahal (for months), Juna Kunada (4 days before)", "Yellow"],
    ["Seepage or wet patches at the toe or bench face", "Jayant 2008 (seepage at the base)", "Yellow"],
    ["Cracks widening or lengthening since the last shift", "Juna Kunada: 1-2 mm to 50 mm in 3 days", "Orange"],
    ["Bulging or heave at the toe", "MKD-III (toe heave); Juna Kunada (floor heaved 15-20 m)", "Orange"],
    ["Water ponding behind a rib, berm or toe", "Bharatpur 2019 (water behind the rib)", "Orange"],
    ["Scarp formed / step in the crest, rocks sliding off the face", "Rajmahal: deep crack and sliding rocks 3.5 h before", "Red"],
    ["Visible movement of the dump mass or toe pushing onto the haul road", "Rajmahal; Juna Kunada (movement sensed, crews withdrawn)", "Red"],
], columns=["Sign", "Seen before these failures", "Level"])

METHODS = pd.DataFrame([
    ["Design FOS by limit equilibrium (Slide2 / Geoslope)", "One condition, no monitoring", "Physics behind the AI FOS (T1)"],
    ["Numerical FOS (RS2, FLAC)", "Slow, one run per case", "Real correction of the AI"],
    ["ML / ANN FOS prediction", "FOS only, no action", "AI FOS re-checked at every reading; water limits"],
    ["Stability rating / hazard class", "Qualitative", "Consequence step, field checklist"],
    ["Prism / radar velocity limits", "Need site calibration", "T3, cross-checked on real failures"],
    ["Inverse velocity (time to failure)", "Only after acceleration starts", "T4"],
    ["Rainfall thresholds", "Not tied to the dump", "T2, linked to FOS through pore pressure"],
    ["Back-analysis of failures", "One case at a time", "Sets the FOS, velocity and rain levels"],
], columns=["Earlier method", "Limitation alone", "Role in this TARP"])

SENSORS = ["Total station + prisms", "Slope stability radar (SSR)", "GNSS monitors", "Crack meters / extensometers",
           "Vibrating-wire piezometers", "Rain gauge (or IMD station)", "UAV / drone survey", "InSAR (Sentinel-1)"]

REFS = [
    ("Statutory and DGMS", [
        ("Coal Mines Regulations 2017, Reg. 106 (spoil banks) and Reg. 108, Govt. of India.", None),
        ("DGMS. Circular on the scientific study under Reg. 106(2) of CMR 2017, para 3.2 (minimum FOS) and para 4.1 "
         "(monitoring and TARP trigger points). Circular number and date to be added.", None),
        ("DGMS Tech. (S&T) Circular No. 02 of 2010, 06.07.2010. Design, control and monitoring of pit and dump slopes "
         "in opencast mines.", "https://elibrarywcl.wordpress.com/wp-content/uploads/2015/02/dgms-tech-st-circular-no-02-of-2010-pit-and-dump-slopes-in-opencast-mines.pdf"),
        ("DGMS Tech. Circular No. 02 of 2020, 09.01.2020. Guidelines for systematic monitoring of slopes (contents "
         "from a study-notes summary).", "https://onlineminingexam.graphy.com/blog/dgms-tech-circular-no-02-of-2020"),
        ("DGMS (2017) Alert: Juna Kunada OC, WCL, 01-12-2017.", "https://www.dgms.gov.in/writereaddata/UploadFile/Alert_Juna_Kunada_OC_WCL_DO_01-12-2017.pdf"),
        ("DGMS (2019) Alert: Bharatpur OCP, MCL, 23-07-2019.", "https://dgms.gov.in/writereaddata/UploadFile/Bharatpur%20OCP-FA-230719-alert.pdf"),
        ("Ministry of Coal (2024) Annual Report 2023-24, ch. 14 Safety in coal mines.", "https://coal.gov.in/sites/default/files/2024-07/chap14AnnualReport2024en2.pdf"),
        ("Ministry of Coal / PIB (2022) Safety measures in coal mines.", "https://www.coal.gov.in/sites/default/files/2022-01/PIB1776761.pdf"),
    ]),
    ("Indian failure cases", [
        ("Dash A.K. (2019) Current Science 117(2):304-308 - slope-failure accidents in Indian opencast mines, "
         "Table 1 (DGMS data) and a dragline-dump case study.", "https://i-scholar.in/index.php/CURS/article/download/187064/174214"),
        ("Mongabay (Feb 2017) Grief, anger and fear in the aftermath of a deadly coal mining disaster in Jharkhand state, India.",
         "https://news.mongabay.com/2017/02/grief-anger-and-fear-in-the-aftermath-of-a-deadly-coal-mining-disaster-in-jharkhand-state-india/"),
        ("Business Standard / IANS (2017) Lalmatia was not fit for mining: DGMS.", "https://business-standard.com/amp/article/news-ians/lalmatia-was-not-fit-for-mining-dgms-117011300587_1.html"),
        ("India Environment Portal (2013) Odisha coal dump crash toll rises to 14.", "https://indiaenvironmentportal.org.in/news-clippings/odisha-coal-dump-crash-toll-rises-to-14"),
        ("Deccan Herald. Landslide in coal mine in Chhattisgarh, one worker trapped (Kusmunda).", "https://www.deccanherald.com/amp/story/india%2Flandslide-in-coal-mine-in-chhattisgarh-one-worker-trapped-864840.html"),
        ("Shende et al. (2025) Phys. Chem. Earth 141:104091 - forensic investigation of a failed OB dump, central India.",
         "https://www.sciencedirect.com/science/article/abs/pii/S1474706525002414"),
        ("Poulsen B. et al. (2014) Mine overburden dump failure: a case study. Geotech. Geol. Eng. 32:297-309.",
         "https://link.springer.com/article/10.1007/s10706-013-9714-7"),
        ("Sharma S., Roy I. (2015) Slope failure of waste rock dump at Jayant opencast mine, India. Int. J. Appl. "
         "Eng. Res. 10(13):33006-33012.", None),
        ("Kainthola A., Verma D., Gupte S.S., Singh T.N. (2011) A coal mine dump stability analysis - a case study. "
         "Geomaterials 1:1-13.", "https://www.scirp.org/html/4675.html"),
        ("Jhanwar J.C. (2010) Investigation into slope failure in an opencast coal mine of Wardha Valley coalfield. "
         "Indian Mining & Engg. J. 49(4):36-40.", "https://cimfr.csircentral.net/743/"),
        ("Bishwal R.M. (2019) List of major coal mine overburden dump failures in India (1995-2019). Mendeley Data "
         "v2, doi:10.17632/ptcbv7cngh.2 (files not public).", "https://data.mendeley.com/datasets/ptcbv7cngh/2"),
    ]),
    ("Monitoring, early warning and TARP", [
        ("Carla T. et al. (2017) Guidelines on the use of inverse velocity method as a tool for setting alarm "
         "thresholds and forecasting landslides and structure collapses. Landslides 14(2):517-534.",
         "https://flore.unifi.it/handle/2158/1044054"),
        ("Ryan T.M., Call R.D. (1992) Applications of rock mass monitoring for stability assessment of pit slope "
         "failure. Rock Mechanics (33rd US Symposium).", "https://www.cnitucson.com/publications/1992_Ryan_520-Rk%20mass%20monitoring%20for%20stability.pdf"),
        ("Carla T., Intrieri E., Farina P., Casagli N. (2017) A new method to identify impending failure in rock "
         "slopes. Int. J. Rock Mech. Min. Sci. 93:76-81 (78 Australian coal-mine failures).",
         "https://flore.unifi.it/bitstream/2158/1073065/1/Carla%cc%80%20et%20al%20IJRM%26MS%202017.pdf"),
        ("Dick G.J., Eberhardt E., Stead D., Rose N.D. (2013) Early detection of impending slope failure in open pit "
         "mines using spatial and temporal analysis of real aperture radar measurements. Slope Stability 2013, ACG.",
         "https://papers.acg.uwa.edu.au/d/1308_66_Dick/66_Dick.pdf"),
        ("Intrieri E., Carla T., Gigli G. (2019) Forecasting the time of failure of landslides at slope-scale: a "
         "literature review. Earth-Science Reviews 193:333-349.",
         "https://flore.unifi.it/retrieve/e398c37e-4b5f-179a-e053-3705fe0a4cff/Intrieri%20et%20al%20ESR%202019.pdf"),
        ("Fukuzono T. (1985) A new method for predicting the failure time of a slope. 4th Int. Conf. Field Workshop "
         "on Landslides, Tokyo, 145-150.", None),
        ("Dick G.J. et al. (2015) Development of an early-warning time-of-failure analysis methodology for open-pit "
         "mine slopes utilizing ground-based slope stability radar. Can. Geotech. J. 52:515-529.",
         "https://www.eoas.ubc.ca/personal/erik/e-papers/15EE_CanGeotechJ-SlopeRadar.pdf"),
        ("Bakken K.M., Chapin G.K., Abrahams M.G. (2020) Trigger action response plan development and optimisation "
         "at the Bingham Canyon Mine. Slope Stability 2020, ACG, Perth, 177-190.",
         "https://papers.acg.uwa.edu.au/d/2025_07_Chapin/07_Chapin.pdf"),
        ("Magoon G., Yang D., Herbert T., Sowa M., Schimann H. (2024) Operational slope stability risk management for "
         "large open pits at the Mount Milligan Mine - a case study. Slope Stability 2024.",
         "https://www.knightpiesold.com/sites/en/assets/File/SS%202024%20-%20Operational%20Slope%20Stability%20Risk%20Management.pdf"),
        ("Shellam & Coggan (2020) Slope Stability 2020 (ACG) - SSR velocity and acceleration trends for deformation TARPs.",
         "https://papers.acg.uwa.edu.au/d/2025_10_Shellam/10_Shellam.pdf"),
        ("Masood M.M., Verma T., Raju G.Y. (2025) Development of a filtered inverse velocity method analyser. J. SAIMM "
         "125(8):477-484.", "https://www.scielo.org.za/pdf/jsaimm/v125n8/10.pdf"),
        ("Masood, Verma & Murmu (2023) ICETE 2023 - use of radar in assessing mine slope stability, Kusmunda.",
         "https://www.atlantis-press.com/article/125994195.pdf"),
        ("Mirmazloumi et al. (2023) Bull. Eng. Geol. Environ. 82:374 - InSAR time series and LSTM for early warning "
         "at mining sites.", "https://link.springer.com/content/pdf/10.1007/s10064-023-03388-w.pdf"),
        ("Farina et al. (2020) Slope Stability 2020 (ACG) - rainfall-threshold early warning system, Grasberg.",
         "https://flore.unifi.it/bitstream/2158/1217193/1/Farina%20et%20al%20ACG%202020.pdf"),
        ("WSP Canada (2023) 2022 Annual Facility Performance Report - Coal Reject Spoils, Fording River (Teck Coal).",
         "https://www.teck.com/media/2022-AFPR-FRO-Coal-Reject-Spoils.pdf"),
        ("Golder Associates & Hungr O. (1995) Runout characteristics of debris from dump failures in mountainous "
         "terrain, Stage 2. BC Mine Waste Rock Pile Research Committee.", "https://www2.gov.bc.ca/assets/gov/farming-natural-resources-and-industry/mineral-exploration-mining/documents/permitting/geotechnical-information/mined_rock_and_overburden_piles_stage_2_feb_1995.pdf"),
        ("Dwinagara B. et al. (2024) Int. J. GEOMATE 27(123):54-64 - back analysis of a rainfall-induced coal waste "
         "dump failure, East Kalimantan.", "https://geomatejournal.com/geomate/article/download/4618/3515/9006"),
        ("Chand & Koner (2023) J. Mining Environ. 14(4) - internal dump stability with UAV-based 3D modelling, Raniganj.", "https://jme.shahroodut.ac.ir/article_2900_53bae7cf957d816026b23afccfce415d.pdf"),
        ("Chand et al. (2024) Sadhana 49:277 - dump slope change detection with UAV photogrammetry.", None),
    ]),
    ("Dump stability and FOS engine", [
        ("NIT Rourkela (2025) Safety and feasibility study of filling of fly ash in eight opencast mines of SECL, "
         "Project C3/23/MN/049 (and scientific-study notes on slope-movement bands).", None),
        ("Chourasia K., Ravi S., Khare R.N. (2025) Jagannathpur OCP dump optimisation. Research Square rs-8013159.", None),
        ("Sahoo A.K., Tripathy D.P., Jayanthu S. (2025) Sci. Rep. 15:40985 and Mendeley dataset.", None),
        ("Singh U.C. et al. (2022) Indian Geotech. J. 52:735-741 - scattered soft soil in coal OB dumps (FLAC3D).",
         "https://link.springer.com/article/10.1007/s40098-022-00596-1"),
        ("Geete et al. (2023) J. Mines Metals Fuels 71(9) - review of OB dump slope stability analysis.",
         "https://informaticsjournals.co.in/index.php/jmmf/article/download/35438/22858/69257"),
        ("Nayak, Dash & Dewangan (2021) J. Mines Metals Fuels 69(7) - rainfall and groundwater effects on a coal "
         "waste dump.", "https://informaticsjournals.co.in/index.php/jmmf/article/view/28542"),
        ("Sahu, Mohanty & Jayanthu (2023) 10th Asian Mining Congress - dump slope stability rating and risk assessment.",
         "https://link.springer.com/chapter/10.1007/978-3-031-46966-4_7"),
        ("Bishop A.W. (1955) The use of the slip circle in the stability analysis of slopes. Geotechnique 5:7-17.", None),
        ("Bishop A.W., Morgenstern N. (1960) Stability coefficients for earth slopes. Geotechnique 10:129-150.", None),
        ("Read J., Stacey P. (2009) Guidelines for Open Pit Slope Design. CSIRO.", None),
        ("India Meteorological Department. Rainfall intensity classes (heavy 64.5-115.5, very heavy 115.6-204.4, "
         "extremely heavy >= 204.5 mm in 24 h).", None),
    ]),
]
