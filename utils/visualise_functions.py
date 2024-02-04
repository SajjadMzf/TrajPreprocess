import cv2

def plot_vehicle(image, centre, dimension, v_color, text, t_color, circle = False, font_scale=1):
    top_left = (int(centre[0]-dimension[0]/2), int(centre[1]-dimension[1]/2))
    bot_left = (int(centre[0]-dimension[0]/2), int(centre[1]+dimension[1]/2))
    bot_right = (int(centre[0]+dimension[0]/2), int(centre[1]+dimension[1]/2))
    if circle == True:
        image = cv2.circle(image, centre, radius = 10, color = v_color, thickness = -1)
    else:
        image = cv2.rectangle(image,top_left, bot_right,color=v_color, thickness = -1)
        image = cv2.putText(image, text, bot_left, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color = t_color, thickness=3*font_scale )
    return image