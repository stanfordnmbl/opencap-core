# This function returns the IDs of the sessions and trials from the lab
# validation. We collected this dataset before developing the web application, 
# and therefore manually entered the ID of each session and trial here.

def getData(sessionName, trialIdx=[]):
    data = {}
    
    # %% Subject 2
    # 1st session
    data['subject2_0'] = {}
    data['subject2_0']['session_id'] = 'b08c621c-234e-4a6b-bcde-eda784e3bcb4'
    data['subject2_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam1', 'Cam4', 'Cam2'], '2-cameras': ['Cam1', 'Cam4']}
    data['subject2_0']['trials'] = {}
    data['subject2_0']['trials'][0] = {"id": "a57ffaac-bdcc-4786-8acf-c0edd92f7109", "intrinsicsFinalFolder": 'Deployed_720_240fps', "extrinsicsTrial": True} # extrinsics
    data['subject2_0']['trials'][2] = {"id": "7027b0cb-e290-4bc9-9fbd-d59b3c8c1b81", "intrinsicsFinalFolder": 'Deployed_720_240fps', 'scaleModel': True} # static1
    data['subject2_0']['trials'][3] = {"id": "89c98148-5dbc-41ce-b983-807db01e3392", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # STS1
    data['subject2_0']['trials'][5] = {"id": "db011831-5b1e-42e3-bd08-3551d12c3617", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # STSweakLegs1
    data['subject2_0']['trials'][6] = {"id": "13c7eeaf-8ab6-4214-ac28-148d3579a4b4", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # squats1
    data['subject2_0']['trials'][7] = {"id": "ba7cfb33-354f-4d68-8ce2-e753e1c10c14", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # squatsAsym1
    data['subject2_0']['trials'][8] = {"id": "d14150de-b9db-430d-85fa-99f1d8d5754c", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJ1
    data['subject2_0']['trials'][9] = {"id": "163174f7-9913-4c79-bda3-cb6551349965", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJ2
    data['subject2_0']['trials'][10] = {"id": "592067e8-78dc-4e3a-a0dd-b1cb197dfaaa", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJ3
    data['subject2_0']['trials'][13] = {"id": "79e3db5c-e433-40c1-b957-45479b510fc9", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJAsym1
    data['subject2_0']['trials'][16] = {"id": "5cadae71-4db0-49f7-a581-bbcf0ca65a17", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJAsym4
    data['subject2_0']['trials'][17] = {"id": "f9e95ded-016c-4859-b1eb-84a4706a49e7", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJAsym5
    # 2nd session (gait trials)
    data['subject2_1'] = {}
    data['subject2_1']['session_id'] = 'b08c621c-234e-4a6b-bcde-eda784e3bcb4'
    data['subject2_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam1', 'Cam4', 'Cam3'], '2-cameras': ['Cam1', 'Cam4']}
    data['subject2_1']['trials'] = {}    
    data['subject2_1']['trials'][0] = {"id": "6368b864-254b-4ea4-bdc5-bc22cef34a94", "intrinsicsFinalFolder": 'Deployed_720_240fps', "extrinsicsTrial": True} # extrinsics
    data['subject2_1']['trials'][2] = {"id": "198881de-42b4-4b46-9e84-9c7ec38fc252", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walking1
    data['subject2_1']['trials'][3] = {"id": "4c56dcd6-6741-4f5e-bcf2-bf8a778ab7a8", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walking2
    data['subject2_1']['trials'][4] = {"id": "7c7e31c2-9e0d-4a23-bded-e8b66049ae1c", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walking3
    data['subject2_1']['trials'][7] = {"id": "8ba0a98f-904c-4138-a8f7-fa1217240aa1", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walkingTS1
    data['subject2_1']['trials'][8] = {"id": "4dc52094-3462-40a6-a785-26875ededed1", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walkingTS2
    data['subject2_1']['trials'][10] = {"id": "5ed94c2e-d860-4ec1-991b-59bc5ce31ece", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walkingTS4
    
    # %% Subject 3
    # 1st session
    data['subject3_0'] = {}
    data['subject3_0']['session_id'] = 'af6a837a-f11a-4399-ab06-a3996a6c70ab'
    data['subject3_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam2', 'Cam3', 'Cam0'], '2-cameras': ['Cam2', 'Cam3']}
    data['subject3_0']['trials'] = {}   
    data['subject3_0']['trials'][0] = {"id": "73346506-e82a-457d-896e-efea2e7d290f", "intrinsicsFinalFolder": 'Deployed_720_240fps', "extrinsicsTrial": True} # extrinsics
    data['subject3_0']['trials'][2] = {"id": "f7d8a34d-03ff-42cd-8f9d-732e86a02533", "intrinsicsFinalFolder": 'Deployed_720_240fps', 'scaleModel': True} # static1
    data['subject3_0']['trials'][3] = {"id": "28aa71cc-347d-4382-a316-abed6a23f159", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # STS1
    data['subject3_0']['trials'][5] = {"id": "43a6c17b-39b4-4de6-ac55-dd65839dc2db", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # STSweakLegs1
    data['subject3_0']['trials'][6] = {"id": "902c400f-ce9c-4f44-9116-40f54317ace6", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # squats1
    data['subject3_0']['trials'][7] = {"id": "65466823-f9e2-468a-a2a8-88c05a6b439b", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # squatsAsym1
    data['subject3_0']['trials'][8] = {"id": "75b01f64-4405-4ac8-816d-a3a3f85b3ba5", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJ1
    data['subject3_0']['trials'][9] = {"id": "394e3245-cb12-4b4d-a741-ef6032de5b79", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJ2
    data['subject3_0']['trials'][11] = {"id": "b0b5ed09-08b7-47d5-9349-8232b56f03c2", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJ4
    data['subject3_0']['trials'][13] = {"id": "f242c16b-c4d6-43d0-835b-658e3df989dd", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJAsym1
    data['subject3_0']['trials'][14] = {"id": "c1574ba5-ff49-47a3-9ae5-2f77d13716bb", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJAsym2
    data['subject3_0']['trials'][16] = {"id": "260573e1-b099-4d8b-a9d6-1b2058f19fe5", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # DJAsym4
    # 2nd session (gait trials)
    data['subject3_1'] = {}
    data['subject3_1']['session_id'] = 'af6a837a-f11a-4399-ab06-a3996a6c70ab'
    data['subject3_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam0', 'Cam1', 'Cam4'], '2-cameras': ['Cam0', 'Cam1']}
    data['subject3_1']['trials'] = {}
    data['subject3_1']['trials'][0] = {"id": "68af81bf-b635-4213-979f-ab3a51d9ffba", "intrinsicsFinalFolder": 'Deployed_720_240fps', "extrinsicsTrial": True} # extrinsics
    data['subject3_1']['trials'][2] = {"id": "0d9e4b14-1590-4ff6-99cc-6f217639e34e", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walking1
    data['subject3_1']['trials'][3] = {"id": "ba7e5aab-b1c6-4590-8ec5-2629f6907e59", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walking2
    data['subject3_1']['trials'][4] = {"id": "ea91fa96-14f2-4727-8bd3-bb418938ce09", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walking3
    data['subject3_1']['trials'][8] = {"id": "60b94a9a-43ea-4636-80fb-72d134a186d6", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walkingTS2
    data['subject3_1']['trials'][9] = {"id": "090c8e78-2951-4f4e-b179-7e6b013b43d7", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walkingTS3
    data['subject3_1']['trials'][10] = {"id": "849635d4-32be-4651-9775-59eaabf8ecf6", "intrinsicsFinalFolder": 'Deployed_720_240fps'} # walkingTS4
    
    # %% Subject 4
    # 1st session
    data['subject4_0'] = {}
    data['subject4_0']['session_id'] = '33b0021f-7032-41bb-bb00-e772a4d6fd1c'
    data['subject4_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam1', 'Cam4', 'Cam0'], '2-cameras': ['Cam1', 'Cam4']}
    data['subject4_0']['trials'] = {}
    data['subject4_0']['trials'][0] = {"id": "3b660454-c3c1-48b6-bd2b-b9270f33b20f", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject4_0']['trials'][2] = {"id": "82355490-67e7-4a7e-9bec-87ee54795719", "intrinsicsFinalFolder": 'Deployed_720_60fps', 'scaleModel': True} # static1
    data['subject4_0']['trials'][3] = {"id": "c9588c39-d639-4151-b4bf-782b2124e93a", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STS1
    data['subject4_0']['trials'][5] = {"id": "fb9ee3bc-59c1-47df-9837-c92c6ef721f3", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STSweakLegs1
    data['subject4_0']['trials'][6] = {"id": "80bd187b-9066-43ae-83b0-70121eb25271", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squats1
    data['subject4_0']['trials'][7] = {"id": "0cb1048f-2010-49be-853a-4c509699adac", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squatsAsym1
    data['subject4_0']['trials'][8] = {"id": "c01ce1f3-cb38-4e54-9204-fa8ab218424f", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ1
    data['subject4_0']['trials'][9] = {"id": "ddb3ddcf-bb59-4ea4-9610-b17db3ff0f0e", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ2
    data['subject4_0']['trials'][10] = {"id": "7c4259f2-3f84-4778-a4cf-734a692661e4", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ3
    data['subject4_0']['trials'][13] = {"id": "e9ba9011-6c69-4fba-a8cb-ff008cab6e40", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym1
    data['subject4_0']['trials'][14] = {"id": "78e2b2d7-3a12-4b1b-bb98-1adadb431506", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym2
    data['subject4_0']['trials'][15] = {"id": "a7ec1965-edf6-43ed-baf7-d3042ea1c61b", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym3
    # 2nd session (gait trials)
    data['subject4_1'] = {}
    data['subject4_1']['session_id'] = '33b0021f-7032-41bb-bb00-e772a4d6fd1c'
    data['subject4_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam3', 'Cam4', 'Cam1'], '2-cameras': ['Cam3', 'Cam4']}
    data['subject4_1']['trials'] = {}
    data['subject4_1']['trials'][0] = {"id": "ac7475ec-2492-4cfb-b14f-45a365e07d96", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject4_1']['trials'][2] = {"id": "0eaaeb83-dba0-4e52-8a9b-fccb344483fe", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking1
    data['subject4_1']['trials'][3] = {"id": "4dad4c6d-a2c7-49a0-8adf-be6d0cb956c6", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking2
    data['subject4_1']['trials'][5] = {"id": "fda6154f-f408-437d-8c4c-a8bcef0bde44", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking4
    data['subject4_1']['trials'][7] = {"id": "ee18cce4-da53-4cc0-8a42-db284f07be6b", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS1 
    data['subject4_1']['trials'][8] = {"id": "017ca355-205e-492b-80ed-128f142bc61a", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS2
    data['subject4_1']['trials'][9] = {"id": "cfea4fd6-d98d-4a61-a322-9537f3518a42", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS3
    
    # %% Subject 5
    # 1st session
    data['subject5_0'] = {}
    data['subject5_0']['session_id'] = '3a262b91-ff8f-4d1e-8273-cdc16bf54045'
    data['subject5_0']['camera_setup'] = {'5-cameras': ['all'],'3-cameras': ['Cam2', 'Cam4', 'Cam3'], '2-cameras': ['Cam2', 'Cam4']}
    data['subject5_0']['trials'] = {}
    data['subject5_0']['trials'][0] = {"id": "8f14e907-7718-4516-b3e1-d90b3fe4ca19", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject5_0']['trials'][2] = {"id": "93465dc3-b2d2-435f-bdb1-85b1ba23e563", "intrinsicsFinalFolder": 'Deployed_720_60fps', 'scaleModel': True} # static1
    data['subject5_0']['trials'][4] = {"id": "f20b6afe-cc7f-44df-81ac-7a5b735bb2a1", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STS1
    data['subject5_0']['trials'][6] = {"id": "e9751325-b38c-47ba-af16-af9effaaf2b4", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STSweakLegs1
    data['subject5_0']['trials'][7] = {"id": "71a61863-f2b7-4a2f-b90b-a1c3b36ec604", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squats1
    data['subject5_0']['trials'][8] = {"id": "47a82a00-ee2c-4222-b812-d4ad05a6d012", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squatsAsym1
    data['subject5_0']['trials'][9] = {"id": "1b9aa15e-1771-421d-8912-a09ea5ce1e1b", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ1
    data['subject5_0']['trials'][10] = {"id": "2068bac0-2db7-4f90-b885-640984ba1af9", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ2
    data['subject5_0']['trials'][11] = {"id": "f863e43c-271e-4b15-9cb2-d670733ea97d", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ3
    data['subject5_0']['trials'][14] = {"id": "f3b6c240-8a72-41b9-80a6-8328d2c5d185", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym1
    data['subject5_0']['trials'][15] = {"id": "0409cce6-70bd-4a79-98ca-342e38461487", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym2
    data['subject5_0']['trials'][16] = {"id": "f4d891ad-8082-46d7-b6d2-0173669d19c2", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym3
    # 2nd session (gait trials)
    data['subject5_1'] = {}
    data['subject5_1']['session_id'] = '3a262b91-ff8f-4d1e-8273-cdc16bf54045'
    data['subject5_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam1', 'Cam3', 'Cam4'], '2-cameras': ['Cam1', 'Cam3']}
    data['subject5_1']['trials'] = {}
    data['subject5_1']['trials'][0] = {"id": "614cffd2-41d2-4fd8-9260-a161de5b5246", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject5_1']['trials'][2] = {"id": "01ab1bdc-6c5e-45e6-b8f0-027beb8ca77a", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking1
    data['subject5_1']['trials'][3] = {"id": "332705f7-a742-4a4a-a699-2479fd292026", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking2
    data['subject5_1']['trials'][4] = {"id": "2b3a6f06-9f1e-429b-a9a5-b4594a89275b", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking3
    data['subject5_1']['trials'][7] = {"id": "8716c5c3-2a61-4429-ade9-597a6ad28027", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS1 
    data['subject5_1']['trials'][8] = {"id": "c152b4f4-46c4-4de1-9cfd-5c2e1ce8cf06", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS2
    data['subject5_1']['trials'][9] = {"id": "caacd204-8d79-499b-953a-e2db7ad8a6ad", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS3
    
    # %% Subject 6
    # 1st session
    data['subject6_0'] = {}
    data['subject6_0']['session_id'] = '7766f365-9610-41fd-b4d3-3b6e6f7ee41b'
    data['subject6_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam0', 'Cam4', 'Cam3'], '2-cameras': ['Cam0', 'Cam4']}
    data['subject6_0']['trials'] = {}
    data['subject6_0']['trials'][0] = {"id": "cf2608c3-d822-43da-a156-3bee542a6f72", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject6_0']['trials'][2] = {"id": "184908e0-e730-424c-b22d-0d3867d079a1", "intrinsicsFinalFolder": 'Deployed_720_60fps', 'scaleModel': True} # static1
    data['subject6_0']['trials'][4] = {"id": "8bdb5d5c-529f-4806-8490-6800984fef14", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STS1
    data['subject6_0']['trials'][6] = {"id": "a03c4cf9-e5bc-46e7-8f07-4cf5d3f6054a", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STSweakLegs1
    data['subject6_0']['trials'][7] = {"id": "f63f2ce2-1d96-4c65-9cff-58f465588d30", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squats1
    data['subject6_0']['trials'][8] = {"id": "cdd015b3-b55e-4fac-aef0-85c84845b6e2", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squatsAsym1
    data['subject6_0']['trials'][9] = {"id": "97cac360-a19f-4b4f-98e2-bdda57cc319a", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ1
    data['subject6_0']['trials'][10] = {"id": "62e83bf2-f367-4e91-91c8-952fa83cdd23", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ2
    data['subject6_0']['trials'][11] = {"id": "bb529d69-6120-4729-a5d0-de367f275157", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ3
    data['subject6_0']['trials'][14] = {"id": "65121aa2-1bbe-4f68-8809-b8a94b7ff01d", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym1
    data['subject6_0']['trials'][15] = {"id": "8a4a1b02-da59-4e28-99dd-b4b8560abf72", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym2
    data['subject6_0']['trials'][16] = {"id": "174bba9a-7064-4df9-b2d2-d6edb8764950", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym3
    # 2nd session (gait trials)
    data['subject6_1'] = {}
    data['subject6_1']['session_id'] = '7766f365-9610-41fd-b4d3-3b6e6f7ee41b'
    data['subject6_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam3', 'Cam4', 'Cam0'], '2-cameras': ['Cam3', 'Cam4']}
    data['subject6_1']['trials'] = {}
    data['subject6_1']['trials'][0] = {"id": "adad95f5-8f86-49bd-bedf-4cfec32fc90d", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject6_1']['trials'][2] = {"id": "13c35f9a-0970-4b5e-b2a8-49c5db377e54", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking1
    data['subject6_1']['trials'][3] = {"id": "da0bb884-9e09-4315-b231-99b7f62c4e18", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking2
    data['subject6_1']['trials'][4] = {"id": "ac16c071-3e71-43ab-92c9-52fc1be743ea", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking3
    data['subject6_1']['trials'][7] = {"id": "aa21855e-fdce-4dae-93a6-6e1a4cd5e409", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS1 
    data['subject6_1']['trials'][8] = {"id": "295158d4-e97e-4cad-b33d-a309699f7d75", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS2
    data['subject6_1']['trials'][9] = {"id": "ab64c203-ebb9-4aff-a4c7-700f45ea20a7", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS3
    
    # %% Subject 7
    # 1st session
    data['subject7_0'] = {}
    data['subject7_0']['session_id'] = '4d4c7231-9537-498b-a865-9ea6187a47cc'
    data['subject7_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam0', 'Cam2', 'Cam4'], '2-cameras': ['Cam0', 'Cam2']}
    data['subject7_0']['trials'] = {}
    data['subject7_0']['trials'][0] = {"id": "b8de62b5-f17b-46bb-a8bd-bcf8d891d1bb", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject7_0']['trials'][2] = {"id": "31dc42d8-d4f8-4c01-ad07-b3c2ac46ef20", "intrinsicsFinalFolder": 'Deployed_720_60fps', 'scaleModel': True} # static1
    data['subject7_0']['trials'][4] = {"id": "5233701f-301f-4763-a7c4-b385ef194a7d", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STS1
    data['subject7_0']['trials'][6] = {"id": "bb600434-f93b-4407-a600-9a96b8420372", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STSweakLegs1
    data['subject7_0']['trials'][7] = {"id": "c2acbd1b-b604-41be-830a-99decf095f85", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squats1
    data['subject7_0']['trials'][8] = {"id": "dffe3708-db5b-4bf1-a4bc-43ab842c5fb6", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squatsAsym1
    data['subject7_0']['trials'][10] = {"id": "a308a059-9e84-4b72-ae2c-268d2a29d40b", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ2
    data['subject7_0']['trials'][11] = {"id": "1d572371-5072-4a59-b8cc-12a5c940eabb", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ3
    data['subject7_0']['trials'][12] = {"id": "a2909772-93ae-4dba-8e46-12e9fb6f8390", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ4
    data['subject7_0']['trials'][14] = {"id": "503ac3ef-19f7-4f6a-bbfd-10b181a21788", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym1
    data['subject7_0']['trials'][15] = {"id": "8ff946f2-ccd3-4b46-ae7f-feff0ed8eb99", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym2
    data['subject7_0']['trials'][16] = {"id": "6179c32f-73fd-4979-84b6-afde1ea814fa", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym3
    # 2nd session (gait trials)
    data['subject7_1'] = {}
    data['subject7_1']['session_id'] = '4d4c7231-9537-498b-a865-9ea6187a47cc'
    data['subject7_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam1', 'Cam2', 'Cam3'], '2-cameras': ['Cam1', 'Cam2']}
    data['subject7_1']['trials'] = {}
    data['subject7_1']['trials'][0] = {"id": "f766fd16-f5aa-4066-92a5-451ae087b962", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject7_1']['trials'][2] = {"id": "c37b171e-799c-432e-ae6f-5097b68cc487", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking1
    data['subject7_1']['trials'][3] = {"id": "10ff2d30-1604-4324-8ac7-ecc1795b1f61", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking2
    data['subject7_1']['trials'][4] = {"id": "90503fcb-5de1-4ac0-a06b-98bbfe71ec50", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking3
    data['subject7_1']['trials'][7] = {"id": "103f3419-d446-458d-b039-b71fb23eabed", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS1 
    data['subject7_1']['trials'][8] = {"id": "858d2634-f90c-44dd-9ff2-afb07bb6459e", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS2
    data['subject7_1']['trials'][9] = {"id": "151786f5-c862-4d67-bdd0-fc4dcedf070c", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS3
    
    # %% Subject 8
    # 1st session
    data['subject8_0'] = {}
    data['subject8_0']['session_id'] = 'ce1fb77c-d497-4df0-81b5-c38ec2ee6585'
    data['subject8_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam0', 'Cam4', 'Cam2'], '2-cameras': ['Cam0', 'Cam4']}
    data['subject8_0']['trials'] = {}
    data['subject8_0']['trials'][0] = {"id": "0a4c7330-1d93-4495-83db-701479a078ec", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject8_0']['trials'][2] = {"id": "a5f9a725-072e-4cc9-a057-a4cde308e8fd", "intrinsicsFinalFolder": 'Deployed_720_60fps', 'scaleModel': True} # static1
    data['subject8_0']['trials'][4] = {"id": "347a90e8-b49a-408b-935f-e8b593916606", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STS1
    data['subject8_0']['trials'][6] = {"id": "c8324066-d4d9-4e4c-9eee-1bedf2db3b55", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STSweakLegs1
    data['subject8_0']['trials'][7] = {"id": "a703138a-f15c-4fd0-9a54-325a54096fcf", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squats1
    data['subject8_0']['trials'][8] = {"id": "25ab79bc-5ec5-44b5-a0a4-c7cda49be0c6", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squatsAsym1
    data['subject8_0']['trials'][9] = {"id": "23cf85ae-eb5f-4472-99f8-6d1a1565d66f", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ1
    data['subject8_0']['trials'][10] = {"id": "d9821044-7b50-44d0-b7fc-b7cb13792159", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ2
    data['subject8_0']['trials'][11] = {"id": "17e7b06b-b64e-484d-bfc7-ec9c601f652e", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ3
    data['subject8_0']['trials'][14] = {"id": "be0445b3-5e08-4f1f-86f8-bf9d795a1b02", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym1
    data['subject8_0']['trials'][15] = {"id": "893b2fea-72d2-4994-8e27-5874662f901f", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym2
    data['subject8_0']['trials'][16] = {"id": "398af567-2e7d-4973-8a2b-d0a7f25e22cc", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym3
    # 2nd session (gait trials)
    data['subject8_1'] = {}
    data['subject8_1']['session_id'] = 'ce1fb77c-d497-4df0-81b5-c38ec2ee6585'
    data['subject8_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam1', 'Cam2', 'Cam3'], '2-cameras': ['Cam1', 'Cam2']}
    data['subject8_1']['trials'] = {}
    data['subject8_1']['trials'][0] = {"id": "d82e1ac4-3e05-4f1e-a729-3a703b0e5237", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject8_1']['trials'][2] = {"id": "0a6f62c7-ed09-43d4-b5ff-7fa8f43fcca5", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking1
    data['subject8_1']['trials'][3] = {"id": "2eeaf665-f3c6-4327-ae5d-ffd230c962e7", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking2
    data['subject8_1']['trials'][4] = {"id": "cc0a5e73-b1be-43ca-a9fb-eb6211d21263", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking3
    data['subject8_1']['trials'][7] = {"id": "07b6c390-2e79-4242-9f0e-48bcbc04e549", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS1 
    data['subject8_1']['trials'][8] = {"id": "aeb2c8f5-e1db-46dc-9387-d069ba52de74", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS2
    data['subject8_1']['trials'][9] = {"id": "262b8c02-4f49-4c70-bc86-3e137d15a1f9", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS3
    
    # %% Subject 9
    # 1st session
    data['subject9_0'] = {}
    data['subject9_0']['session_id'] = '8c194b0b-c832-442c-93be-3f3a0006d7dc'
    data['subject9_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam2', 'Cam4', 'Cam0'], '2-cameras': ['Cam2', 'Cam4']}
    data['subject9_0']['trials'] = {}
    data['subject9_0']['trials'][0] = {"id": "f9d8f305-eff1-4329-b5ed-a1ac5ba18c83", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject9_0']['trials'][2] = {"id": "c364ee13-5b6b-4cd0-88d8-77c3e64ad74d", "intrinsicsFinalFolder": 'Deployed_720_60fps', 'scaleModel': True} # static1
    data['subject9_0']['trials'][4] = {"id": "220d7fd9-8622-4fcf-a38c-df9927c6808c", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STS1
    data['subject9_0']['trials'][6] = {"id": "48922cb3-ac64-4147-9cbc-1c63473b6da0", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STSweakLegs1
    data['subject9_0']['trials'][7] = {"id": "dff15f36-e99d-420d-969a-1d009df26b53", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squats1
    data['subject9_0']['trials'][8] = {"id": "2481a120-7ca8-4d66-a8bb-a94b01cdb140", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squatsAsym1
    data['subject9_0']['trials'][9] = {"id": "92465af1-9112-4b37-8c57-acb3c6e91bfc", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ1
    data['subject9_0']['trials'][10] = {"id": "460a97fd-1960-42ca-8306-31b708762a3f", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ2
    data['subject9_0']['trials'][11] = {"id": "bab0f0df-2897-4538-95f8-7c550f7624de", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ3
    data['subject9_0']['trials'][14] = {"id": "1da3fa8e-2798-4221-987c-dfba0c0a0d47", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym1
    data['subject9_0']['trials'][15] = {"id": "0e798bec-58e3-468d-ae10-512c35b1dcc5", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym2
    data['subject9_0']['trials'][16] = {"id": "febcdd3e-3f52-4c1c-93a9-027df494ce3b", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym3
    # 2nd session (gait trials)
    data['subject9_1'] = {}
    data['subject9_1']['session_id'] = '8c194b0b-c832-442c-93be-3f3a0006d7dc'
    data['subject9_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam0', 'Cam4', 'Cam2'], '2-cameras': ['Cam0', 'Cam4']}
    data['subject9_1']['trials'] = {}
    data['subject9_1']['trials'][0] = {"id": "70ccfe2d-9e95-42e7-8143-80716d793847", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject9_1']['trials'][2] = {"id": "3fcc68f8-8e29-4b95-a1d1-59e30c44efb0", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking1
    data['subject9_1']['trials'][3] = {"id": "a31315c3-e028-4c4d-a6b5-1d94c60153a4", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking2
    data['subject9_1']['trials'][4] = {"id": "81ea808a-26da-43b3-a6aa-22b44e0b4807", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking3
    data['subject9_1']['trials'][7] = {"id": "c504e5ae-6cb5-41ea-8ed3-d50b4a3aed82", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS1 
    data['subject9_1']['trials'][8] = {"id": "87dc4c8a-d798-40a7-8199-84003947fabd", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS2
    data['subject9_1']['trials'][9] = {"id": "cf3231e0-7dfe-4dc1-a6a1-7b03ae5ca3c3", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS3
    
    # %% Subject 10
    # 1st session
    data['subject10_0'] = {}
    data['subject10_0']['session_id'] = '1cecfa11-c3ce-44e7-8bc8-f5f10314a6cc'
    data['subject10_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam3', 'Cam4', 'Cam1'], '2-cameras': ['Cam3', 'Cam4']}
    data['subject10_0']['trials'] = {}
    data['subject10_0']['trials'][0] = {"id": "f360afc0-5ac2-47cc-9b31-9a5792429952", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject10_0']['trials'][2] = {"id": "f3c270c9-8bda-4b5a-a6fe-9dc9c485d0b3", "intrinsicsFinalFolder": 'Deployed_720_60fps', 'scaleModel': True} # static1
    data['subject10_0']['trials'][4] = {"id": "cc87bc63-9814-4970-a47e-ab42218a3b52", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STS1
    data['subject10_0']['trials'][6] = {"id": "03fe22e4-6975-4e00-9618-f0f47b328d6b", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STSweakLegs1
    data['subject10_0']['trials'][7] = {"id": "b12cbb5f-0651-4a0b-b3bd-28ef6e8a62fd", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squats1
    data['subject10_0']['trials'][8] = {"id": "2d406abb-c064-4276-bac0-f8a9499f44d7", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squatsAsym1
    data['subject10_0']['trials'][9] = {"id": "e116e964-71c4-4924-a298-75672e24e4cd", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ1
    data['subject10_0']['trials'][10] = {"id": "83d8846f-e956-4362-ac56-4f41b5f99763", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ2
    data['subject10_0']['trials'][11] = {"id": "8d4688d8-752a-4cb4-bd9b-3a3244a6a078", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ3
    data['subject10_0']['trials'][14] = {"id": "6eb4f36b-12de-4b84-b30e-604ad99a2809", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym1
    data['subject10_0']['trials'][15] = {"id": "ec73be70-9378-4987-8930-315b353e5545", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym2
    data['subject10_0']['trials'][16] = {"id": "444cce46-5d29-4b84-8e56-5e252805fc10", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym3
    # 2nd session (gait trials)
    data['subject10_1'] = {}
    data['subject10_1']['session_id'] = '1cecfa11-c3ce-44e7-8bc8-f5f10314a6cc'
    data['subject10_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam1', 'Cam4', 'Cam3'], '2-cameras': ['Cam1', 'Cam4']}
    data['subject10_1']['trials'] = {}
    data['subject10_1']['trials'][0] = {"id": "e9fc4546-b26a-47e4-b339-6b32f2411236", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject10_1']['trials'][2] = {"id": "0c57ab46-a84b-4a3c-ba81-4470ccbb86e2", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking1
    data['subject10_1']['trials'][3] = {"id": "90af677b-284c-4091-8f29-8bcceea3b3a7", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking2
    data['subject10_1']['trials'][4] = {"id": "ad3c637c-be1f-4abd-b03b-f9760230a62c", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking3
    data['subject10_1']['trials'][7] = {"id": "cdd527eb-db9e-43f7-bace-654c82e39e13", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS1 
    data['subject10_1']['trials'][8] = {"id": "4d30bdfc-bd11-44da-a2c8-9ef9d685c936", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS2
    data['subject10_1']['trials'][9] = {"id": "d323c621-5b96-4891-9503-af2ee1d08ddd", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS3
    
    # %% Subject 11
    # 1st session
    data['subject11_0'] = {}
    data['subject11_0']['session_id'] = '8195493a-8eae-4003-87b5-740a64567bb2'
    data['subject11_0']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam1', 'Cam2', 'Cam3'], '2-cameras': ['Cam1', 'Cam2']}
    data['subject11_0']['trials'] = {}
    data['subject11_0']['trials'][0] = {"id": "7ee31a3b-47c6-4d7b-a7f4-0867af1ea457", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject11_0']['trials'][2] = {"id": "4a8ae9b1-4b1a-43db-899d-1b25ad24f71d", "intrinsicsFinalFolder": 'Deployed_720_60fps', 'scaleModel': True} # static1
    data['subject11_0']['trials'][4] = {"id": "be1ac547-153f-4cd7-86f7-cd472470f8bc", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STS1
    data['subject11_0']['trials'][6] = {"id": "58d7a449-3884-4544-aa40-d0d5a486f9d3", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # STSweakLegs1
    data['subject11_0']['trials'][7] = {"id": "40a4018d-bb35-491b-945d-3d5dec67c3ce", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squats1
    data['subject11_0']['trials'][8] = {"id": "7b51f532-72e6-4c2e-bea9-6fb48f171c1b", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # squatsAsym1
    data['subject11_0']['trials'][9] = {"id": "8b0749ce-b3ab-4e6a-91e3-71c65e1734c0", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ1
    data['subject11_0']['trials'][12] = {"id": "dc3103b8-e769-496c-9190-957952dccb8c", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ4
    data['subject11_0']['trials'][13] = {"id": "5a1ac175-cfcb-498e-af14-d1ad37f3b464", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJ5
    data['subject11_0']['trials'][16] = {"id": "167b6e93-94d3-4afa-aae0-76903982f40f", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym3
    data['subject11_0']['trials'][17] = {"id": "7245d4a5-9576-4d60-b07a-f3471816ed75", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym4
    data['subject11_0']['trials'][18] = {"id": "b6890400-8fab-4115-b5c7-91efbd6fd38c", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # DJAsym5
    # 2nd session (gait trials)
    data['subject11_1'] = {}
    data['subject11_1']['session_id'] = '8195493a-8eae-4003-87b5-740a64567bb2'
    data['subject11_1']['camera_setup'] = {'5-cameras': ['all'], '3-cameras': ['Cam2', 'Cam3', 'Cam0'], '2-cameras': ['Cam2', 'Cam3']}
    data['subject11_1']['trials'] = {}
    data['subject11_1']['trials'][0] = {"id": "010b31a3-6931-44da-b152-b5f2486be092", "extrinsicsTrial": True, "intrinsicsFinalFolder": 'Deployed_720_60fps'} # extrinsics
    data['subject11_1']['trials'][3] = {"id": "299abc65-00fa-444b-ba93-33d054b68a89", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking2
    data['subject11_1']['trials'][4] = {"id": "3cb77421-2841-41e8-82fc-063edcdb7ad5", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking3
    data['subject11_1']['trials'][5] = {"id": "482885ff-1c48-4ab8-b1bc-c299a6cd90c7", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walking4
    data['subject11_1']['trials'][7] = {"id": "35326ae2-7b75-4c44-9ee8-c4b5014f521c", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS1 
    data['subject11_1']['trials'][8] = {"id": "c2cdfde7-2a62-44b3-a5bc-b5a7772eb3ac", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS2
    data['subject11_1']['trials'][9] = {"id": "740460ca-f828-42a6-81bf-f9485f78dd27", "intrinsicsFinalFolder": 'Deployed_720_60fps'} # walkingTS3

    
    # %% Return data    
    if trialIdx:
        dataOut = {}
        dataOut[sessionName] = {}
        dataOut[sessionName]['session_id'] = data[sessionName]['session_id']
        if 'camera_setup' in data[sessionName]:
            dataOut[sessionName]['camera_setup'] = data[sessionName]['camera_setup']
        dataOut[sessionName]['trials'] = {}
        for count, i in enumerate(trialIdx):
            dataOut[sessionName]['trials'][count] = data[sessionName]['trials'][i]
        return dataOut[sessionName]
            
    else:
        return data[sessionName]
