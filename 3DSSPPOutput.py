import pandas as pd
import numpy as np

class SSPPOutput:
    def __init__(self):
        self.set_default_header()
        pass

    def set_default_header(self):
        default_header = ['Analyst', 'Company Name', 'Units', 'Task Name', 'Gender', 'Height', 'Weight', 'Summary Start', 'L5/S1 Compression', 'L4/L5 Compression', 'Minimum Wrist Percentile', 'Minimum Elbow Percentile', 'Minimum Shoulder Percentile', 'Minimum Torso Percentile', 'Minimum Hip Percentile', 'Minimum Knee Percentile', 'Minimum Ankle Percentile', 'Coefficient of Friction', 'Left Load Fraction', 'Balance Status', 'Strength Capability Start', 'Right Wrist Flexion', 'Right Wrist Deviation', 'Right Forearm Rotation', 'Right Elbow Flexion', 'Right Humeral Rotation', 'Right Shoulder Rotation', 'Right Shoulder Abduction', 'Right Hip Flexion', 'Right Knee Flexion', 'Right Ankle Flexion', 'Left Wrist Flexion', 'Left Wrist Deviation', 'Left Forearm Rotation', 'Left Elbow Flexion', 'Left Humeral Rotation', 'Left Shoulder Rotation', 'Left Shoulder Abduction', 'Left Hip Flexion', 'Left Knee Flexion', 'Left Ankle Flexion', 'Torso Flexion', 'Torso Lateral Bending', 'Torso Rotation', 'Low Back Start', 'L5/S1 Compression', 'L5/S1 Compression SD', 'Sagittal Shear', 'Frontal Shear', 'Total L4/L5 Compression', 'Anterior L4/L5 Shear', 'Lateral L4/L5 Shear', 'Right Erector Force Magnitude', 'Right Erector Shear', 'Right Erector Force X', 'Right Erector Force Y', 'Right Erector Force Z', 'Right Rectus Force Magnitude', 'Right Rectus Shear', 'Right Rectus Force X', 'Right Rectus Force Y', 'Right Rectus Force Z', 'Right Internal Force Magnitude', 'Right Internal Shear', 'Right Internal Force X', 'Right Internal Force Y', 'Right Internal Force Z', 'Right External Force Magnitude', 'Right External Shear', 'Right External Force X', 'Right External Force Y', 'Right External Force Z', 'Right Latissimus Force Magnitude', 'Right Latissimus Shear', 'Right Latissimus Force X', 'Right Latissimus Force Y', 'Right Latissimus Force Z', 'Left Erector Force Magnitude', 'Left Erector Shear', 'Left Erector Force X', 'Left Erector Force Y', 'Left Erector Force Z', 'Left Rectus Force Magnitude', 'Left Rectus Shear', 'Left Rectus Force X', 'Left Rectus Force Y', 'Left Rectus Force Z', 'Left Internal Force Magnitude', 'Left Internal Shear', 'Left Internal Force X', 'Left Internal Force Y', 'Left Internal Force Z', 'Left External Force Magnitude', 'Left External Shear', 'Left External Force X', 'Left External Force Y', 'Left External Force Z', 'Left Latissimus Force Magnitude', 'Left Latissimus Shear', 'Left Latissimus Force X', 'Left Latissimus Force Y', 'Left Latissimus Force Z', 'Fatigue Start', 'Right Wrist Flexion Fifth', 'Right Wrist Flexion Fiftieth', 'Right Wrist Flexion Nintieth', 'Right Wrist Deviation Fifth', 'Right Wrist Deviation Fiftieth', 'Right Wrist Deviation Nintieth', 'Right Forearm Rotation Fifth', 'Right Forearm Rotation Fiftieth', 'Right Forearm Rotation Nintieth', 'Right Elbow Flexion Fifth', 'Right Elbow Flexion Fiftieth', 'Right Elbow Flexion Nintieth', 'Right Humeral Rotation Fifth', 'Right Humeral Rotation Fiftieth', 'Right Humeral Rotation Nintieth', 'Right Shoulder Rotation Fifth', 'Right Shoulder Rotation Fiftieth', 'Right Shoulder Rotation Nintieth', 'Right Shoulder Abduction Fifth', 'Right Shoulder Abduction Fiftieth', 'Right Shoulder Abduction Nintieth', 'Right Hip Flexion Fifth', 'Right Hip Flexion Fiftieth', 'Right Hip Flexion Nintieth', 'Right Knee Flexion Fifth', 'Right Knee Flexion Fiftieth', 'Right Knee Flexion Nintieth', 'Right Ankle Flexion Fifth', 'Right Ankle Flexion Fiftieth', 'Right Ankle Flexion Nintieth', 'Left Wrist Flexion Fifth', 'Left Wrist Flexion Fiftieth', 'Left Wrist Flexion Nintieth', 'Left Wrist Deviation Fifth', 'Left Wrist Deviation Fiftieth', 'Left Wrist Deviation Nintieth', 'Left Forearm Rotation Fifth', 'Left Forearm Rotation Fiftieth', 'Left Forearm Rotation Nintieth', 'Left Elbow Flexion Fifth', 'Left Elbow Flexion Fiftieth', 'Left Elbow Flexion Nintieth', 'Left Humeral Rotation Fifth', 'Left Humeral Rotation Fiftieth', 'Left Humeral Rotation Nintieth', 'Left Shoulder Rotation Fifth', 'Left Shoulder Rotation Fiftieth', 'Left Shoulder Rotation Nintieth', 'Left Shoulder Abduction Fifth', 'Left Shoulder Abduction Fiftieth', 'Left Shoulder Abduction Nintieth', 'Left Hip Flexion Fifth', 'Left Hip Flexion Fiftieth', 'Left Hip Flexion Nintieth', 'Left Knee Flexion Fifth', 'Left Knee Flexion Fiftieth', 'Left Knee Flexion Nintieth', 'Left Ankle Flexion Fifth', 'Left Ankle Flexion Fiftieth', 'Left Ankle Flexion Nintieth', 'Torso Flexion Fifth', 'Torso Flexion Fiftieth', 'Torso Flexion Nintieth', 'Torso Lateral Bending Fifth', 'Torso Lateral Bending Fiftieth', 'Torso Lateral Bending Nintieth', 'Torso Rotation Fifth', 'Torso Rotation Fiftieth', 'Torso Rotation Nintieth', 'Balance Start', 'COG X', 'COG Y', 'COP X', 'COP Y', 'Stability', 'Left load', 'Right load', 'Hand Forces Start', 'Right Force Magnitude', 'Right Vertical Angle', 'Right Horizontal Angle', 'Left Force Magnitude', 'Left Vertical Angle', 'Left Horizontal Angle', 'Segment Angles Start', 'Right Vertical Hand Angle', 'Right Horizontal Hand Angle', 'Right Hand Rotation Angle', 'Right Vertical Forearm Angle', 'Right Horizontal Forearm Angle', 'Right Vertical Upper Arm Angle', 'Right Horizontal Upper Arm Angle', 'Right Vertical Clavicle Angle', 'Right Horizontal Clavicle Angle', 'Right Vertical Upper Leg Angle', 'Right Horizontal Upper Leg Angle', 'Right Vertical Lower Leg Angle', 'Right Horizontal Lower Leg Angle', 'Right Vertical Foot Angle', 'Right Horizontal Foot Angle', 'Left Vertical Hand Angle', 'Left Horizontal Hand Angle', 'Left Hand Rotation Angle', 'Left Vertical Forearm Angle', 'Left Horizontal Forearm Angle', 'Left Vertical Upper Arm Angle', 'Left Horizontal Upper Arm Angle', 'Left Vertical Clavicle Angle', 'Left Horizontal Clavicle Angle', 'Left Vertical Upper Leg Angle', 'Left Horizontal Upper Leg Angle', 'Left Vertical Lower Leg Angle', 'Left Horizontal Lower Leg Angle', 'Left Vertical Foot Angle', 'Left Horizontal Foot Angle', 'Head Lateral Bending Angle', 'Head Flexion Angle', 'Head Axial Rotation Angle', 'Trunk Lateral Bending Angle', 'Trunk Flexion Angle', 'Trunk Axial Rotation Angle', 'Pelvis Lateral Bending Angle', 'Pelvis Flexion Angle', 'Pelvis Axial Rotation Angle', 'Posture Angles Start', 'Right Hand Flexion', 'Right Hand Deviation', 'Right Forearm Rotation', 'Right Elbow Included', 'Right Shoulder Vertical', 'Right Shoulder Horizontal', 'Right Humeral Rotation', 'Right Hip Included', 'Right Hip Vertical', 'Right Hip Horizontal', 'Right Femoral Rotation', 'Right Lower Leg Rotation', 'Right Knee Included', 'Right Ankle Included', 'Left Hand Flexion', 'Left Hand Deviation', 'Left Forearm Rotation', 'Left Elbow Included', 'Left Shoulder Vertical', 'Left Shoulder Horizontal', 'Left Humeral Rotation', 'Left Hip Included', 'Left Hip Vertical', 'Left Hip Horizontal', 'Left Femoral Rotation', 'Left Lower Leg Rotation', 'Left Knee Included', 'Left Ankle Included', 'Head Flexion Angle', 'Head Axial Rotation Angle', 'Head Lateral Bending Angle', 'Trunk Flexion From L5/S1', 'Adjusted Trunk Axial Rotation', 'Adjusted Trunk Lateral Bending', 'Pelvis Flexion', 'Pelvis Axial Rotation Angle', 'Pelvis Lateral Bending Angle', 'L5S1 Tilt Angle', 'Joint Locations Start', 'Right Hand X', 'Right Hand Y', 'Right Hand Z', 'Right Wrist X', 'Right Wrist Y', 'Right Wrist Z', 'Right Elbow X', 'Right Elbow Y', 'Right Elbow Z', 'Right Shoulder X', 'Right Shoulder Y', 'Right Shoulder Z', 'Right Hip X', 'Right Hip Y', 'Right Hip Z', 'Right IT X', 'Right IT Y', 'Right IT Z', 'Right Knee X', 'Right Knee Y', 'Right Knee Z', 'Right Ankle X', 'Right Ankle Y', 'Right Ankle Z', 'Right Heel X', 'Right Heel Y', 'Right Heel Z', 'Right Foot Center X', 'Right Foot Center Y', 'Right Foot Center Z', 'Right Ball of Foot X', 'Right Ball of Foot Y', 'Right Ball of Foot Z', 'Left Hand X', 'Left Hand Y', 'Left Hand Z', 'Left Wrist X', 'Left Wrist Y', 'Left Wrist Z', 'Left Elbow X', 'Left Elbow Y', 'Left Elbow Z', 'Left Shoulder X', 'Left Shoulder Y', 'Left Shoulder Z', 'Left Hip X', 'Left Hip Y', 'Left Hip Z', 'Left IT X', 'Left IT Y', 'Left IT Z', 'Left Knee X', 'Left Knee Y', 'Left Knee Z', 'Left Ankle X', 'Left Ankle Y', 'Left Ankle Z', 'Left Heel X', 'Left Heel Y', 'Left Heel Z', 'Left Foot Center X', 'Left Foot Center Y', 'Left Foot Center Z', 'Left Ball of Foot X', 'Left Ball of Foot Y', 'Left Ball of Foot Z', 'Tragion X', 'Tragion Y', 'Tragion Z', 'Nasion X', 'Nasion Y', 'Nasion Z', 'Top of Neck X', 'Top of Neck Y', 'Top of Neck Z', 'C7T1 X', 'C7T1 Y', 'C7T1 Z', 'SCJ X', 'SCJ Y', 'SCJ Z', 'L5S1 X', 'L5S1 Y', 'L5S1 Z', 'Center of Hips X', 'Center of Hips Y', 'Center of Hips Z', 'Center of ITs X', 'Center of ITs Y', 'Center of ITs Z', 'Center of Ankles X', 'Center of Ankles Y', 'Center of Ankles Z', 'Center of Balls of Feet X', 'Center of Balls of Feet Y', 'Center of Balls of Feet Z', 'Joint Forces Start', 'Right Hand X', 'Right Hand Y', 'Right Hand Z', 'Right Wrist X', 'Right Wrist Y', 'Right Wrist Z', 'Right Elbow X', 'Right Elbow Y', 'Right Elbow Z', 'Right Shoulder X', 'Right Shoulder Y', 'Right Shoulder Z', 'Right Hip X', 'Right Hip Y', 'Right Hip Z', 'Right IT X', 'Right IT Y', 'Right IT Z', 'Right Knee X', 'Right Knee Y', 'Right Knee Z', 'Right Ankle X', 'Right Ankle Y', 'Right Ankle Z', 'Right Heel X', 'Right Heel Y', 'Right Heel Z', 'Right Foot Center X', 'Right Foot Center Y', 'Right Foot Center Z', 'Right Ball of Foot X', 'Right Ball of Foot Y', 'Right Ball of Foot Z', 'Left Hand X', 'Left Hand Y', 'Left Hand Z', 'Left Wrist X', 'Left Wrist Y', 'Left Wrist Z', 'Left Elbow X', 'Left Elbow Y', 'Left Elbow Z', 'Left Shoulder X', 'Left Shoulder Y', 'Left Shoulder Z', 'Left Hip X', 'Left Hip Y', 'Left Hip Z', 'Left IT X', 'Left IT Y', 'Left IT Z', 'Left Knee X', 'Left Knee Y', 'Left Knee Z', 'Left Ankle X', 'Left Ankle Y', 'Left Ankle Z', 'Left Heel X', 'Left Heel Y', 'Left Heel Z', 'Left Foot Center X', 'Left Foot Center Y', 'Left Foot Center Z', 'Left Ball of Foot X', 'Left Ball of Foot Y', 'Left Ball of Foot Z', 'Tragion X', 'Tragion Y', 'Tragion Z', 'Nasion X', 'Nasion Y', 'Nasion Z', 'Top of Neck X', 'Top of Neck Y', 'Top of Neck Z', 'C7T1 X', 'C7T1 Y', 'C7T1 Z', 'SCJ X', 'SCJ Y', 'SCJ Z', 'L5S1 X', 'L5S1 Y', 'L5S1 Z', 'Center of Hips X', 'Center of Hips Y', 'Center of Hips Z', 'Center of ITs X', 'Center of ITs Y', 'Center of ITs Z', 'Center of Ankles X', 'Center of Ankles Y', 'Center of Ankles Z', 'Center of Balls of Feet X', 'Center of Balls of Feet Y', 'Center of Balls of Feet Z', 'Right Forward Seat X', 'Right Forward Seat Y', 'Right Forward Seat Z', 'Left Forward Seat X', 'Left Forward Seat Y', 'Left Forward Seat Z', 'Seat Back X', 'Seat Back Y', 'Seat Back Z', 'Joint Moments Start', 'Right Hand X', 'Right Hand Y', 'Right Hand Z', 'Right Wrist X', 'Right Wrist Y', 'Right Wrist Z', 'Right Elbow X', 'Right Elbow Y', 'Right Elbow Z', 'Right Shoulder X', 'Right Shoulder Y', 'Right Shoulder Z', 'Right Hip X', 'Right Hip Y', 'Right Hip Z', 'Right IT X', 'Right IT Y', 'Right IT Z', 'Right Knee X', 'Right Knee Y', 'Right Knee Z', 'Right Ankle X', 'Right Ankle Y', 'Right Ankle Z', 'Right Heel X', 'Right Heel Y', 'Right Heel Z', 'Right Foot Center X', 'Right Foot Center Y', 'Right Foot Center Z', 'Right Ball of Foot X', 'Right Ball of Foot Y', 'Right Ball of Foot Z', 'Left Hand X', 'Left Hand Y', 'Left Hand Z', 'Left Wrist X', 'Left Wrist Y', 'Left Wrist Z', 'Left Elbow X', 'Left Elbow Y', 'Left Elbow Z', 'Left Shoulder X', 'Left Shoulder Y', 'Left Shoulder Z', 'Left Hip X', 'Left Hip Y', 'Left Hip Z', 'Left IT X', 'Left IT Y', 'Left IT Z', 'Left Knee X', 'Left Knee Y', 'Left Knee Z', 'Left Ankle X', 'Left Ankle Y', 'Left Ankle Z', 'Left Heel X', 'Left Heel Y', 'Left Heel Z', 'Left Foot Center X', 'Left Foot Center Y', 'Left Foot Center Z', 'Left Ball of Foot X', 'Left Ball of Foot Y', 'Left Ball of Foot Z', 'Tragion X', 'Tragion Y', 'Tragion Z', 'Nasion X', 'Nasion Y', 'Nasion Z', 'Top of Neck X', 'Top of Neck Y', 'Top of Neck Z', 'C7T1 X', 'C7T1 Y', 'C7T1 Z', 'SCJ X', 'SCJ Y', 'SCJ Z', 'L5S1 X', 'L5S1 Y', 'L5S1 Z', 'Center of Hips X', 'Center of Hips Y', 'Center of Hips Z', 'Center of ITs X', 'Center of ITs Y', 'Center of ITs Z', 'Center of Ankles X', 'Center of Ankles Y', 'Center of Ankles Z', 'Center of Balls of Feet X', 'Center of Balls of Feet Y', 'Center of Balls of Feet Z']
        # modify header to make them unique
        header_category = 'Info'
        header_cat_dict = {header_category: []}
        unique_header = []
        for index, header_item in enumerate(default_header):
            if header_item.endswith('Start'):
                header_category = header_item.replace(' Start', '')
                header_cat_dict[header_category] = []
            else:
                header_item = header_category + ' - ' + header_item
                header_cat_dict[header_category].append(header_item)
            unique_header.append(header_item)

        self.header = unique_header
        self.header_category = header_cat_dict
        # self.info_header = self.header[:7]
        # self.summary_header = self.header[8:20]
        # self.strength_capability_header = self.header[21:44]
        # self.low_back_header = self.header[45:102]
        # self.fatigue_header = self.header[103:172]
        # self.balance_header = self.header[173:180]
        # self.hand_forces_header = self.header[181:187]
        # self.segment_angles_header = self.header[188:227]
        # self.posture_angles_header = self.header[228:266]
        # self.joint_locations_header = self.header[267:363]
        # self.joint_forces_header = self.header[364:469]
        # self.joint_moments_header = self.header[470:566]

    def load_exp_file(self, file=r'F:\wen_storage\3D SSPP\smpl_3DSSPP_batch_0.exp'):
        '''
        load as csv text and read into dict with subheaders
        '''
        # load csv file
        df = pd.read_csv(file, header=None)
        # set header to self.header
        df.columns = self.header
        self.df = df

    def get_category(self, category='Info'):
        return self.df[self.header_category[category]]

    @property
    def category(self):
        return list(self.header_category.keys())

    def get_




if __name__ == "__main__":
    exp_file = r'F:\wen_storage\3D SSPP\smpl_3DSSPP_batch_all.exp'
    result = SSPPOutput()
    result.load_exp_file(exp_file)
    a = result.get_category('Info')
    result.category
    result.header
    result.df['Info - Task Name'][200]

