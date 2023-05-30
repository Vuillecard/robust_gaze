# render lighting and accessory
def render(face_filename, acc_filename, ambient=False, mode='face', direction=((-1,0,0),)):
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # load face mesh object
    face_mesh = load_objs_as_meshes([face_filename], device=device)
    #### replace face texture
    texture_map = face_mesh.textures.maps_padded()
    if mode=='acc' and ambient:
        new_map = torch.zeros_like(texture_map)   
    else:
        new_map = torch.ones_like(texture_map)   
    new_texture = TexturesUV(maps=new_map, 
                             faces_uvs=face_mesh.textures.faces_uvs_padded(), 
                             verts_uvs=face_mesh.textures.verts_uvs_padded())
    face_mesh.textures = new_texture

    if acc_filename is not None:
        # load acc mesh object
        acc_mesh = load_objs_as_meshes([acc_filename], device=device)
        # concat face and acc mesh
        mesh = join_meshes_as_scene([face_mesh, acc_mesh])
        print(mesh.verts_list()[0].shape)
    else:
        mesh = face_mesh

    # Initialize a camera
    R, T = look_at_view_transform(10, 0, 0)
    T *= -1
    R = R@torch.Tensor([[[1,  0.,  0.],    # rotate by 180 degrees
             [ 0.,  -1.,  0.],
             [ 0.,  0., -1.]]])
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=224, 
        blur_radius=0.0, 
        faces_per_pixel=1)

    # Place a point light in front of the object
    # we use only diffuse lighting as described in the paper changing light 
    if not ambient:
        lights = DirectionalLights(device=device, direction=direction,
                                     ambient_color=((0.0, 0.0, 0.0),),
                                     diffuse_color=((1., 1., 1.),),
                                     specular_color=((1.0, 1.0, 1.0),)
                                     )
    else:
        lights = AmbientLights(device=device)
    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0,0,0))
    if not ambient:
        specular_color = [[0.1,0.1,0.1]]
        if mode=='acc':
            specular_color = [[1,1,1]]
        materials = Materials(
                    device=device,
                    ambient_color = [[1,1,1]],
                    diffuse_color = [[1,1,1]],
                    specular_color= specular_color,
                    shininess = 1
                )
    else:
        materials = Materials(
                    device=device,
                    ambient_color = [[1,1,1]],
                    diffuse_color = [[0,0,0]],
                    specular_color= [[0,0,0]],
                    shininess = 0
                )
    images = renderer(mesh, materials=materials, blend_params=blend_params)
    return images


# main function
def get_render(image_path, face_obj_path, acc_obj_path, direction):
    
    # render accessory
    render_acc_dir = render_new(face_obj_path, acc_obj_path, mode='acc', direction=direction)
    render_acc_amb = render_new(face_obj_path, acc_obj_path, mode='acc', ambient=True)
    light_img_acc = 0.6*render_acc_dir[0,:,:,:3].cpu().numpy() + 0.4*render_acc_amb[0,:,:,:3].cpu().numpy()
    
    # get accessory mask
    mask_acc = render_acc_amb[0,:,:,:3]
    mask_acc = mask_acc.sum(axis=-1)!=0
    mask_acc = mask_acc.cpu().numpy()
    
    # render face
    render_face = render_new(face_obj_path, None, mode='face', direction=direction)
    light_img_face = render_face[0,:,:,:3].cpu().numpy()
    light_img_face_norm = np.linalg.norm(light_img_face, axis=-1)
    
    # get face mask
    mask_face = render_new(face_obj_path, None, ambient=True, mode='face')
    mask_face = mask_face[0,:,:,:3]
    mask_face = mask_face.sum(axis=-1)!=0
    mask_face = mask_face.cpu().numpy()
    
    # read og image
    og_img = cv2.imread(image_path)
    
    # modulate light value of og image
    hls_image = cv2.cvtColor(og_img, cv2.COLOR_BGR2HSV)
    hue, saturation, lightness = cv2.split(hls_image)
    light_img_face_smooth = light_img_face_norm
    num_iters = 1
    for i in range(num_iters):    # smooth light img
        light_img_face_smooth = cv2.GaussianBlur(light_img_face_smooth, (5,5), 0)
    modulated_lightness = lightness * (1 - mask_face) + 0.6*(mask_face * light_img_face_smooth*lightness) + 0.4*(lightness * mask_face)
    modulated_lightness = np.clip(modulated_lightness, 0, 255).astype(np.uint8)
    modulated_hls_image = cv2.merge((hue, saturation, modulated_lightness))
    modulated_image = cv2.cvtColor(modulated_hls_image, cv2.COLOR_HSV2BGR)
    
    # add accessory to modulated image
    if len(mask_acc.shape)==2:
        mask_acc = np.expand_dims(mask_acc, -1)
    modulated_image_scaled = modulated_image / 255.0
    light_img_acc = cv2.cvtColor(light_img_acc, cv2.COLOR_BGR2RGB)
    acc_img = light_img_acc*mask_acc + modulated_image_scaled*(1-mask_acc)
    acc_img = acc_img[...,::-1]
    
    return acc_img
    
    