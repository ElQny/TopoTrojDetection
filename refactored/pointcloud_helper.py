def createSamplePointcloud(pointcloud_c: Dict) -> Dict:
    if (pointcloud_c != None):
        return pointcloud_c;
    else:
        pointcloud_c = torch.randn((1,N,3)) * 0.01