import mediapipe as mp
import numpy as np
import cv2


# Indices for the relevant landmarks
_LEFT_EYE_LANDMARK_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE_LANDMARK_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# (v1, v2, v3) vertex tuples specifying a mesh (in terms of landmark indices) spanning the full face
# (extracted from mediapipe\modules\face_geometry\data\canonical_face_model.obj).
_FACE_MESH = frozenset([
    (223, 52, 224), (46, 225, 53), (191, 80, 183), (154, 155, 157), (123, 50, 117),
    (37, 39, 72), (464, 413, 465),
    (340, 261, 346), (451, 252, 452), (281, 275, 363), (290, 250, 328), (142, 36, 129),
    (77, 90, 96), (105, 104, 63),
    (295, 285, 442), (280, 425, 411), (11, 72, 12), (82, 87, 13), (213, 215, 192),
    (187, 147, 192), (122, 196, 6),
    (403, 316, 404), (161, 246, 163), (5, 195, 51), (207, 205, 187), (306, 292, 307),
    (199, 428, 200),
    (422, 432, 273), (241, 238, 242), (218, 115, 219), (92, 165, 206), (404, 315, 405),
    (255, 249, 339),
    (248, 281, 456), (316, 403, 317), (278, 439, 294), (367, 416, 364), (445, 444, 260),
    (436, 427, 426),
    (320, 321, 307), (248, 195, 281), (201, 200, 83), (431, 262, 395), (51, 45, 5),
    (127, 34, 162), (237, 218, 239),
    (52, 105, 53), (387, 373, 388), (377, 400, 396), (60, 99, 20), (69, 67, 104),
    (130, 226, 25), (331, 358, 279),
    (375, 307, 321), (447, 366, 454), (220, 45, 134), (197, 6, 196), (225, 30, 224),
    (128, 121, 114), (193, 245, 122),
    (273, 335, 422), (214, 210, 212), (461, 462, 458), (44, 1, 45), (435, 288, 401),
    (434, 416, 427), (309, 438, 459),
    (394, 379, 364), (387, 259, 386), (81, 82, 41), (237, 241, 44), (333, 299, 334),
    (360, 279, 420), (293, 334, 283),
    (434, 364, 416), (140, 32, 170), (73, 74, 41), (189, 190, 244), (363, 456, 281),
    (125, 19, 44), (334, 296, 282),
    (257, 258, 386), (207, 214, 216), (405, 314, 406), (146, 43, 91), (464, 453, 463),
    (188, 114, 174), (83, 84, 182),
    (194, 201, 182), (208, 201, 32), (242, 141, 241), (409, 410, 270), (212, 216, 214),
    (183, 62, 191),
    (317, 402, 312), (425, 266, 426), (181, 180, 91), (256, 452, 252), (126, 217, 47),
    (99, 240, 97), (301, 298, 300),
    (411, 427, 416), (312, 268, 13), (6, 168, 122), (222, 223, 28), (289, 305, 455),
    (11, 12, 302), (235, 59, 219),
    (165, 39, 167), (355, 429, 371), (181, 182, 84), (350, 277, 349), (270, 322, 269),
    (156, 139, 143),
    (159, 160, 145), (173, 155, 133), (20, 238, 79), (161, 160, 30), (273, 287, 375),
    (386, 374, 387),
    (303, 271, 304), (450, 348, 449), (111, 117, 31), (308, 415, 324), (174, 196, 188),
    (186, 57, 185), (98, 64, 129),
    (233, 232, 128), (335, 406, 424), (455, 439, 289), (18, 83, 200), (347, 449, 348),
    (415, 407, 310),
    (163, 110, 144), (249, 255, 263), (125, 44, 241), (166, 60, 79), (333, 298, 332),
    (2, 164, 326), (41, 38, 73),
    (219, 166, 218), (199, 175, 428), (411, 416, 376), (66, 69, 105), (466, 467, 388),
    (272, 304, 271),
    (178, 88, 179), (429, 420, 279), (407, 408, 272), (50, 187, 205), (19, 94, 354),
    (106, 91, 43), (187, 192, 207),
    (304, 408, 270), (115, 220, 131), (367, 397, 435), (51, 134, 45), (311, 271, 312),
    (381, 384, 382),
    (47, 100, 126), (1, 44, 19), (71, 21, 139), (408, 306, 409), (204, 106, 202),
    (298, 333, 293), (448, 339, 449),
    (447, 264, 345), (19, 125, 94), (356, 389, 264), (383, 300, 353), (158, 159, 153),
    (57, 186, 212),
    (422, 430, 432), (304, 270, 303), (102, 129, 64), (20, 79, 60), (158, 28, 159),
    (145, 23, 153), (292, 306, 407),
    (236, 174, 198), (137, 93, 177), (242, 20, 99), (379, 394, 378), (299, 337, 296),
    (268, 302, 12), (60, 166, 75),
    (426, 322, 436), (108, 109, 69), (377, 396, 152), (195, 248, 197), (413, 464, 414),
    (66, 65, 107), (79, 218, 166),
    (233, 128, 244), (17, 314, 16), (72, 73, 38), (276, 283, 445), (320, 404, 321),
    (111, 31, 35), (27, 223, 29),
    (11, 302, 0), (296, 334, 299), (274, 275, 1), (104, 103, 68), (50, 205, 101), (326, 370, 2),
    (60, 75, 99),
    (424, 431, 422), (256, 341, 452), (24, 23, 144), (14, 13, 87), (423, 266, 358),
    (404, 320, 403), (307, 325, 320),
    (101, 36, 100), (449, 254, 450), (113, 124, 226), (119, 120, 230), (351, 419, 412),
    (8, 285, 9), (277, 350, 343),
    (83, 18, 84), (337, 299, 338), (294, 331, 278), (141, 94, 125), (3, 51, 195), (42, 183, 80),
    (185, 40, 186),
    (17, 16, 84), (326, 393, 327), (118, 117, 50), (4, 275, 5), (434, 430, 364), (162, 139, 21),
    (262, 431, 418),
    (308, 324, 292), (124, 113, 46), (319, 403, 320), (171, 175, 208), (418, 424, 406),
    (237, 220, 218),
    (302, 268, 303), (189, 193, 221), (300, 293, 276), (445, 342, 276), (178, 81, 88),
    (408, 407, 306), (53, 224, 52),
    (143, 34, 116), (35, 226, 124), (121, 128, 232), (331, 294, 358), (278, 344, 439),
    (246, 247, 33),
    (417, 351, 465), (392, 289, 439), (401, 366, 376), (366, 401, 323), (228, 31, 117),
    (267, 0, 302),
    (288, 435, 397), (170, 149, 140), (394, 364, 430), (452, 350, 451), (254, 449, 339),
    (297, 338, 299),
    (89, 90, 179), (58, 172, 215), (431, 395, 430), (299, 333, 297), (115, 131, 48),
    (107, 108, 66), (391, 269, 322),
    (314, 405, 315), (358, 371, 429), (227, 234, 137), (347, 280, 346), (186, 92, 216),
    (252, 451, 253),
    (449, 347, 448), (243, 244, 190), (57, 212, 43), (113, 247, 225), (215, 213, 177),
    (418, 421, 262),
    (226, 130, 113), (319, 325, 318), (83, 182, 201), (35, 143, 111), (437, 343, 399),
    (103, 104, 67),
    (322, 270, 410), (160, 161, 144), (330, 329, 266), (240, 75, 235), (204, 211, 194),
    (354, 370, 461),
    (446, 342, 359), (398, 382, 384), (229, 228, 118), (21, 71, 54), (140, 176, 171),
    (55, 221, 193), (15, 14, 86),
    (313, 314, 18), (437, 420, 355), (435, 401, 433), (217, 174, 114), (430, 422, 431),
    (235, 64, 240),
    (244, 243, 233), (384, 385, 286), (436, 410, 432), (55, 107, 65), (290, 305, 392),
    (396, 369, 428),
    (214, 192, 135), (34, 127, 227), (106, 204, 182), (4, 5, 45), (155, 112, 133),
    (328, 326, 460), (381, 382, 256),
    (4, 1, 275), (237, 239, 241), (312, 13, 317), (354, 274, 19), (390, 388, 373),
    (325, 292, 324), (328, 462, 326),
    (169, 135, 150), (17, 84, 18), (125, 241, 141), (307, 375, 306), (131, 198, 49),
    (434, 432, 430), (412, 465, 351),
    (90, 91, 180), (276, 353, 300), (41, 42, 81), (295, 282, 296), (286, 414, 384),
    (148, 171, 176), (291, 306, 375),
    (50, 101, 118), (284, 332, 298), (121, 120, 47), (48, 219, 115), (151, 337, 10),
    (63, 68, 70), (279, 360, 278),
    (237, 44, 220), (385, 386, 258), (54, 68, 103), (139, 162, 34), (369, 396, 400),
    (409, 270, 408), (199, 200, 208),
    (303, 269, 302), (174, 236, 196), (76, 77, 62), (271, 303, 268), (310, 272, 311),
    (264, 447, 356),
    (334, 293, 333), (382, 362, 341), (440, 363, 275), (191, 95, 80), (112, 233, 243),
    (363, 440, 360),
    (190, 173, 243), (243, 133, 112), (323, 454, 366), (234, 227, 127), (147, 187, 123),
    (135, 138, 136),
    (61, 185, 57), (319, 320, 325), (86, 85, 15), (336, 9, 285), (137, 123, 227),
    (258, 442, 286), (285, 295, 336),
    (250, 458, 462), (122, 188, 196), (238, 241, 239), (353, 265, 383), (443, 282, 442),
    (141, 242, 97),
    (357, 465, 343), (31, 25, 226), (22, 153, 23), (295, 442, 282), (202, 212, 210),
    (2, 97, 164), (352, 376, 366),
    (43, 202, 106), (442, 258, 443), (298, 301, 284), (257, 259, 443), (205, 206, 36),
    (78, 191, 62), (283, 276, 293),
    (195, 197, 3), (325, 307, 292), (371, 266, 329), (402, 317, 403), (450, 253, 451),
    (221, 222, 56),
    (339, 448, 255), (85, 86, 180), (337, 151, 336), (254, 373, 253), (154, 26, 155),
    (457, 459, 438),
    (329, 349, 277), (93, 137, 234), (210, 211, 202), (173, 190, 157), (42, 41, 74),
    (368, 301, 383), (345, 372, 340),
    (316, 15, 315), (393, 326, 164), (26, 232, 112), (228, 229, 110), (349, 451, 350),
    (175, 171, 152), (95, 96, 88),
    (224, 29, 223), (90, 77, 91), (209, 129, 49), (168, 8, 193), (28, 158, 56), (151, 10, 108),
    (359, 263, 255),
    (415, 310, 324), (291, 287, 409), (136, 150, 135), (148, 152, 171), (67, 69, 109),
    (427, 436, 434),
    (138, 215, 172), (455, 294, 439), (230, 229, 119), (456, 420, 399), (313, 421, 406),
    (277, 355, 329),
    (132, 58, 177), (123, 117, 116), (68, 54, 71), (7, 163, 246), (343, 437, 277),
    (459, 458, 309), (361, 323, 401),
    (9, 336, 151), (259, 387, 260), (240, 99, 75), (50, 123, 187), (344, 360, 440),
    (246, 33, 7), (248, 456, 419),
    (214, 207, 192), (396, 428, 175), (263, 466, 249), (45, 220, 44), (421, 313, 200),
    (315, 404, 316),
    (462, 328, 250), (360, 420, 363), (392, 309, 290), (73, 72, 39), (39, 40, 73),
    (395, 369, 378), (315, 16, 314),
    (466, 263, 467), (175, 152, 396), (324, 318, 325), (374, 380, 253), (244, 245, 189),
    (165, 167, 98),
    (418, 406, 421), (74, 40, 184), (462, 461, 370), (463, 341, 362), (269, 303, 270),
    (209, 49, 198), (87, 86, 14),
    (290, 328, 305), (454, 356, 447), (433, 376, 416), (131, 134, 198), (74, 184, 42),
    (335, 321, 406),
    (199, 208, 175), (427, 411, 425), (402, 318, 311), (355, 277, 437), (414, 286, 413),
    (203, 36, 206),
    (49, 48, 131), (265, 340, 372), (221, 55, 222), (111, 116, 117), (3, 196, 236),
    (168, 417, 8), (345, 352, 447),
    (393, 164, 267), (7, 33, 25), (374, 253, 373), (375, 321, 273), (64, 48, 102),
    (458, 459, 461), (457, 461, 459),
    (181, 91, 182), (134, 131, 220), (27, 159, 28), (260, 388, 467), (341, 256, 382),
    (251, 284, 301),
    (280, 347, 330), (247, 113, 130), (346, 448, 347), (407, 415, 292), (453, 452, 341),
    (128, 114, 245),
    (96, 95, 62), (120, 121, 231), (398, 384, 414), (63, 53, 105), (46, 70, 124),
    (438, 439, 344), (62, 183, 76),
    (206, 216, 92), (180, 181, 85), (275, 274, 440), (309, 392, 438), (362, 398, 463),
    (432, 434, 436),
    (36, 101, 205), (257, 386, 259), (172, 136, 138), (291, 375, 287), (353, 276, 342),
    (165, 92, 39), (163, 7, 110),
    (245, 244, 128), (443, 444, 282), (177, 147, 137), (188, 245, 114), (232, 231, 121),
    (300, 383, 301),
    (390, 373, 339), (86, 87, 179), (179, 180, 86), (388, 260, 387), (65, 222, 55),
    (272, 310, 407), (229, 230, 24),
    (429, 355, 420), (327, 294, 460), (358, 327, 423), (365, 364, 379), (258, 286, 385),
    (0, 164, 37), (79, 239, 218),
    (126, 142, 209), (112, 155, 26), (89, 96, 90), (417, 168, 351), (444, 445, 283),
    (383, 372, 368), (88, 80, 95),
    (444, 443, 259), (279, 278, 331), (119, 118, 101), (250, 290, 309), (194, 32, 201),
    (193, 189, 245),
    (261, 446, 255), (153, 22, 154), (313, 406, 314), (203, 129, 36), (40, 39, 92), (37, 72, 0),
    (225, 46, 113),
    (223, 222, 52), (78, 95, 191), (287, 432, 410), (15, 316, 14), (262, 428, 369),
    (342, 446, 353), (216, 212, 186),
    (5, 281, 195), (226, 35, 31), (359, 467, 263), (429, 279, 358), (215, 177, 58),
    (227, 116, 34), (221, 56, 189),
    (27, 28, 223), (11, 0, 72), (197, 419, 6), (327, 460, 326), (371, 329, 355), (78, 62, 95),
    (171, 208, 140),
    (433, 416, 435), (89, 179, 88), (457, 438, 440), (423, 391, 426), (204, 202, 211),
    (85, 84, 16), (319, 318, 403),
    (143, 35, 156), (380, 385, 381), (37, 167, 39), (190, 189, 56), (106, 182, 91),
    (23, 24, 230), (369, 395, 262),
    (352, 346, 280), (129, 209, 142), (130, 25, 33), (280, 330, 425), (196, 3, 197),
    (163, 144, 161), (246, 161, 247),
    (184, 185, 76), (460, 455, 305), (167, 164, 97), (138, 135, 192), (285, 8, 417),
    (301, 368, 251), (412, 399, 343),
    (81, 178, 82), (382, 398, 362), (351, 6, 419), (30, 247, 161), (441, 442, 285),
    (108, 107, 151), (344, 440, 438),
    (224, 53, 225), (370, 326, 462), (261, 255, 448), (456, 363, 420), (70, 46, 63),
    (124, 156, 35), (20, 242, 238),
    (282, 283, 334), (166, 219, 59), (283, 282, 444), (101, 100, 119), (97, 2, 141),
    (425, 426, 427), (410, 409, 287),
    (102, 49, 129), (231, 232, 22), (71, 70, 68), (340, 346, 345), (134, 51, 236),
    (340, 265, 261), (77, 76, 146),
    (150, 149, 169), (8, 9, 55), (285, 417, 441), (264, 368, 372), (332, 297, 333),
    (169, 210, 135), (26, 22, 232),
    (453, 464, 357), (202, 43, 212), (455, 460, 294), (256, 252, 381), (157, 158, 154),
    (296, 336, 295),
    (344, 278, 360), (117, 118, 228), (4, 45, 1), (29, 30, 160), (59, 75, 166), (280, 411, 352),
    (390, 339, 249),
    (310, 311, 318), (364, 365, 367), (28, 56, 222), (107, 55, 9), (185, 184, 40), (80, 88, 81),
    (43, 146, 57),
    (14, 317, 13), (211, 210, 170), (250, 309, 458), (385, 380, 386), (231, 230, 120),
    (100, 47, 120), (98, 97, 240),
    (145, 153, 159), (371, 358, 266), (180, 179, 90), (388, 390, 466), (236, 198, 134),
    (16, 315, 15),
    (428, 262, 421), (271, 311, 272), (47, 114, 121), (311, 312, 402), (145, 144, 23),
    (372, 345, 264), (99, 97, 242),
    (341, 463, 453), (132, 177, 93), (391, 327, 393), (354, 461, 274), (391, 423, 327),
    (367, 435, 416),
    (29, 224, 30), (87, 82, 178), (173, 157, 155), (9, 151, 107), (376, 352, 411),
    (394, 430, 395), (265, 353, 446),
    (441, 286, 442), (82, 13, 38), (216, 206, 207), (118, 119, 229), (350, 452, 357),
    (80, 81, 42), (209, 198, 126),
    (268, 312, 271), (139, 156, 71), (424, 422, 335), (405, 321, 404), (446, 261, 265),
    (374, 386, 380),
    (253, 450, 254), (114, 47, 217), (188, 122, 245), (133, 243, 173), (1, 19, 274),
    (410, 436, 322), (203, 206, 165),
    (361, 401, 288), (109, 108, 10), (97, 98, 167), (412, 343, 465), (400, 378, 369),
    (16, 15, 85), (259, 260, 444),
    (17, 18, 314), (30, 225, 247), (65, 66, 52), (169, 170, 210), (115, 218, 220),
    (92, 186, 40), (160, 159, 29),
    (130, 33, 247), (252, 253, 380), (123, 137, 147), (267, 269, 393), (318, 324, 310),
    (395, 378, 394),
    (249, 466, 390), (26, 154, 22), (327, 358, 294), (94, 141, 2), (368, 264, 389),
    (289, 392, 305), (96, 62, 77),
    (69, 66, 108), (116, 227, 123), (293, 300, 298), (239, 79, 238), (110, 25, 228),
    (254, 339, 373), (441, 413, 286),
    (217, 126, 198), (24, 110, 229), (359, 255, 446), (74, 73, 40), (260, 467, 445),
    (335, 273, 321), (402, 403, 318),
    (308, 292, 415), (31, 228, 25), (467, 359, 342), (201, 208, 200), (235, 219, 64),
    (193, 122, 168), (3, 236, 51),
    (59, 235, 75), (338, 10, 337), (213, 192, 147), (267, 302, 269), (61, 76, 185),
    (370, 354, 94), (233, 112, 232),
    (463, 414, 464), (0, 267, 164), (397, 367, 365), (48, 64, 219), (144, 145, 160),
    (465, 357, 464), (120, 119, 100),
    (12, 13, 268), (372, 383, 265), (352, 345, 346), (357, 343, 350), (129, 203, 98),
    (391, 393, 269),
    (291, 409, 306), (380, 381, 252), (55, 193, 8), (6, 351, 168), (414, 463, 398),
    (146, 91, 77), (70, 71, 156),
    (116, 111, 143), (423, 426, 266), (24, 144, 110), (38, 12, 72), (266, 425, 330),
    (94, 2, 370), (68, 63, 104),
    (349, 329, 348), (61, 146, 76), (61, 57, 146), (89, 88, 96), (157, 56, 158),
    (257, 443, 258), (439, 438, 392),
    (451, 349, 450), (65, 52, 222), (176, 140, 149), (448, 346, 261), (413, 441, 417),
    (419, 197, 248), (12, 38, 13),
    (34, 143, 139), (25, 110, 7), (27, 29, 159), (32, 194, 211), (105, 52, 66), (22, 23, 231),
    (421, 200, 428),
    (38, 41, 82), (317, 14, 316), (376, 433, 401), (56, 157, 190), (53, 63, 46),
    (453, 357, 452), (373, 387, 374),
    (437, 399, 420), (165, 98, 203), (304, 272, 408), (322, 426, 391), (389, 251, 368),
    (399, 412, 419),
    (84, 85, 181), (366, 447, 352), (184, 76, 183), (211, 170, 32), (142, 100, 36),
    (457, 440, 274), (342, 445, 467),
    (98, 240, 64), (348, 330, 347), (424, 418, 431), (384, 381, 385), (214, 135, 210),
    (178, 179, 87), (156, 124, 70),
    (330, 348, 329), (287, 273, 432), (167, 37, 164), (205, 207, 206), (32, 140, 208),
    (230, 231, 23),
    (348, 450, 349), (183, 42, 184), (336, 296, 337), (18, 200, 313), (147, 177, 213),
    (405, 406, 321),
    (138, 192, 215), (399, 419, 456), (417, 465, 413), (460, 305, 328), (217, 198, 174),
    (153, 154, 158),
    (104, 105, 69), (170, 169, 149), (49, 102, 48), (142, 126, 100), (281, 5, 275),
    (457, 274, 461), (194, 182, 204)
])


def _get_bounding_box_inflation_factor(eye_coordinates, amplification=2, base_inflation=1):
    """
    Calculate and return the factor at which the perimeter of the bounding box of a face should be inflated by. This is calculated
    as a function of the gradient of the line going through the left and right eyes.
    :param eye_coordinates: [right_eye, left_eye] list containing the normalised coordinates of the eyes. Must be
    a list of mediapipe.framework.formats.location_data_pb2.RelativeKeypoint objects.
    :param amplification: The calculated inflation factor will be multiplied by this value. Defaults to 2.
    :param base_inflation: A base inflation to be added to the final inflation factor. defaults to 1 for 100% base inflation.
    :return: The inflation factor. E.g. returns 0.5 to inflate box perimeter by 50%.
    """

    roll_angle = _get_face_roll_angle([eye_coordinates[1].x, 1 - eye_coordinates[1].y], [eye_coordinates[0].x, 1 - eye_coordinates[0].y])
    inflation_factor = np.abs(roll_angle) / 90

    return base_inflation + (inflation_factor * amplification)


def _get_inflated_face_image(image, face_box, inflation):
    """
    Crop and return the located face in the image. The perimeter of the crop is inflated around the center using the provided inflation factor.
    :param image: The image containing the face.
    :param face_box: The bounding box of the detected face. Must be a mediapipe.framework.formats.location_data_pb2.RelativeBoundingBox object.
    :param inflation: The factor by which the bounding box of a face should be inflated. E.g. 0.5 will inflate the box's perimeter by 50% around the centre.
    :return: A sub-image containing only the face.
    """

    width_inflation, height_inflation = face_box.width * inflation, face_box.height * inflation

    return _crop_within_bounds(
        image,
        round((face_box.ymin - height_inflation / 2) * image.shape[0]),                    # top
        round((face_box.ymin + face_box.height + height_inflation / 2) * image.shape[0]),  # bottom
        round((face_box.xmin - width_inflation / 2) * image.shape[1]),                     # left
        round((face_box.xmin + face_box.width + width_inflation / 2) * image.shape[1])     # right
    )


def _get_segmented_face_image(image, mesh, landmarks):
    """
    Set non-face pixels in the specified image to 0 and return it
    :param image: The image containing the face
    :param mesh: A list of (v1, v2, v3) vertex tuples specifying a mesh (in terms of landmark indices) spanning the face in the image.
    :param landmarks: Landmark coordinates for the face in the image. Must be a list of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    :return: An image with non-face pixels set to 0
    """
    mask = np.zeros(np.shape(image), dtype=np.uint8)
    for triangle in mesh:
        mask_vertices = np.ndarray.astype(np.rint(np.array([
            [landmarks[triangle[0]].x * image.shape[1], landmarks[triangle[0]].y * image.shape[0]],
            [landmarks[triangle[1]].x * image.shape[1], landmarks[triangle[1]].y * image.shape[0]],
            [landmarks[triangle[2]].x * image.shape[1], landmarks[triangle[2]].y * image.shape[0]]
        ])), np.int64)
        cv2.fillPoly(mask, [mask_vertices], (255, 255, 255))
    return cv2.bitwise_and(image, mask)


def _get_left_and_right_eye_centres(left_eye_landmarks, right_eye_landmarks):
    """
    Calculate and return the normalised coordinates of the centres of the left and right eyes in the face. The y
    coordinate is converted from a row number to a height value so that the y coordinate increases for points higher up in the image.
    :param left_eye_landmarks: The landmarks for the left eye (left from the perspective of the image, not the scene).
    Must be a list of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    :param right_eye_landmarks: The landmarks for the right eye (right from the perspective of the image, not the scene).
    Must be a list of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    :return: ([l_x, l_y], [r_x, r_y]) tuple of two numpy arrays containing the normalised coordinates of the centres of the left and right eyes respectively.
    """

    left_eye_centre = np.array([
        np.sum([landmark.x for landmark in left_eye_landmarks]) / len(left_eye_landmarks),
        1 - np.sum([landmark.y for landmark in left_eye_landmarks]) / len(left_eye_landmarks)])

    right_eye_centre = np.array([
        np.sum([landmark.x for landmark in right_eye_landmarks]) / len(right_eye_landmarks),
        1 - np.sum([landmark.y for landmark in right_eye_landmarks]) / len(right_eye_landmarks)])

    return left_eye_centre, right_eye_centre


def _get_eyes_midpoint(left_eye_centre, right_eye_centre, image_size):
    """
    Calculate and return the pixel coordinates of the midpoint between the two eyes of a face. All values are rounded to the nearest integer.
    :param left_eye_centre: [x, y] array containing the normalised coordinates of the left eye's centre. The y
    coordinate must be a height value such that the y coordinate increases for points higher up in the image.
    :param right_eye_centre: [x, y] array containing the normalised coordinates of the right eye's centre. The y
    coordinate must be a height value such that the y coordinate increases for points higher up in the image.
    :param image_size: (height, width) tuple containing the dimensions of the image with the face.
    :return: [x, y] integer numpy array containing the pixel coordinates of the midpoint between the left and right eyes.
    """

    return np.ndarray.astype(np.rint(np.array([
        (left_eye_centre[0] + right_eye_centre[0]) * image_size[1] / 2,
        (left_eye_centre[1] + right_eye_centre[1]) * image_size[0] / 2])), np.int64)


def _get_face_roll_angle(left_eye_centre, right_eye_centre):
    """
    Calculate and return the in-plane rotation angle of a face (in degrees) using the centre coordinates of the eyes.
    (Assumes the image has been flipped i.e. the left eye is on the right of the image and vice versa)
    :param left_eye_centre: [x, y] array containing the coordinates of the left eye's centre. The y value must be a
    height value such that it is higher for points higher up in the image.
    :param right_eye_centre: [x, y] array containing the pixel coordinates of the right eye's centre. The y value must be a
    height value such that it is higher for points higher up in the image.
    :return: The roll angle of the face in degrees (Angles with magnitude 90 or below are given as anticlockwise: +ve, clockwise: -ve. Larger angles are only given as clockwise: +ve).
    """

    if right_eye_centre[1] == left_eye_centre[1]:  # 0 or 180 degree roll
        if left_eye_centre[0] >= right_eye_centre[0]:
            return 0
        else:
            return 180

    elif right_eye_centre[0] == left_eye_centre[0]:  # 90 or 270 (-90) degree roll
        if left_eye_centre[1] > right_eye_centre[1]:
            return 90
        else:
            return -90

    else:  # Roll between 0 and 360 degrees excluding 0, 90, and 270
        gradient = (left_eye_centre[1] - right_eye_centre[1]) / (left_eye_centre[0] - right_eye_centre[0])
        if left_eye_centre[0] > right_eye_centre[0]:
            return np.degrees(np.arctan(gradient))  # Roll between 0 and 90 or 270 (-90) and 0
        else:
            return 180 + np.degrees(np.arctan(gradient))  # Roll between 90 and 270 (-90) degrees


def _rotate_landmarks(landmarks, rotation_matrix, image_size):
    """
    Rotate the landmark points using the specified rotation matrix. The new points are rounded to the nearest
    values and converted to integer types.
    :param landmarks: The landmarks to be rotated. Must be a list of
    mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    :param rotation_matrix: 3x2 transformation matrix for rotation around a specified point, by a specified angle.
    :param image_size: (height, width) tuple containing the dimensions of the image containing the landmarks.
    :return: A 2xn integer numpy matrix where n is the number of landmarks. The first row contains the new x values while
    the second row contains the new y values. Each column contains the new coordinates for a landmark such that the first
    column stores the first landmark's new location, and so on.
    """

    return np.ndarray.astype(np.rint(np.matmul(rotation_matrix, np.array([
            np.multiply([landmark.x for landmark in landmarks], image_size[1]),
            np.multiply([landmark.y for landmark in landmarks], image_size[0]),
            np.ones(len(landmarks))]))), np.int64)


def _get_roll_corrected_image_and_landmarks(face_image, face_landmarks):
    """
    Correct the roll of the given face image and landmarks
    :param face_image: The face image to be roll-corrected
    :param face_landmarks: Face landmarks to be roll-corrected. Must be a list of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    :return: A (corrected_face_image, corrected_face_landmarks) tuple
    """
    left_eye_centre, right_eye_centre = _get_left_and_right_eye_centres(
        [face_landmarks[landmark] for landmark in _LEFT_EYE_LANDMARK_INDICES],
        [face_landmarks[landmark] for landmark in _RIGHT_EYE_LANDMARK_INDICES])
    eyes_midpoint = _get_eyes_midpoint(left_eye_centre, right_eye_centre, face_image.shape)

    roll_angle = _get_face_roll_angle(left_eye_centre, right_eye_centre)
    rotation_matrix = cv2.getRotationMatrix2D((round(eyes_midpoint[0]), round(eyes_midpoint[1])), -roll_angle, 1)

    return cv2.warpAffine(face_image, rotation_matrix, (face_image.shape[1], face_image.shape[0])), _rotate_landmarks(face_landmarks, rotation_matrix, face_image.shape)


def _crop_within_bounds(image, top, bottom, left, right):
    """
    Crop the supplied image within the provided boundaries. If a boundary is outside the perimeter of the image, it
    is clipped to the perimeter edge value. All edge parameters are inclusive
    :param image: The image to be cropped.
    :param top: Maximum integer y coordinate of the crop boundary. Must be a row number instead of a height value such
    that it is lower for points higher up in the image.
    :param bottom: Minimum integer y coordinate of the crop boundary. Must be a row number instead of a height value such
    that it is lower for points higher up in the image.
    :param left: Minimum integer x coordinate of the crop boundary.
    :param right: Maximum integer x coordinate of the crop boundary.
    :return: The cropped image.
    """

    if top < 0: top = 0
    elif top >= image.shape[0]: top = image.shape[0] - 1

    if bottom < 0: bottom = 0
    elif bottom >= image.shape[0]: bottom = image.shape[0] - 1

    if left < 0: left = 0
    elif left >= image.shape[1]: left = image.shape[1] - 1

    if right < 0: right = 0
    elif right >= image.shape[1]: right = image.shape[1] - 1

    return image[top:bottom+1, left:right+1]


class FaceCropper:
    """
    Implements the following pipeline for cropping out faces from an image:
        1. Retrieves bounding boxes and approximate eye coordinates for faces in the image, using the mp.solutions.face_detection.FaceDetection network
            - For more about this network, visit https://solutions.mediapipe.dev/face_detection
        2. For faces with in-plane rotation (i.e. roll), the FaceDetection network gives bounding boxes that are much smaller than the face, hence:
            2.1 The approximate roll of a face is found by calculating the angle between the line going through the eye coordinates and the horizontal
            2.2 The bounding boxes are inflated by a factor relative to the roll of the faces, to ensure bounding boxes include the full face under rotation
        3. The inflated bounding boxes are cropped from the image and passed to the mp.solutions.face_mesh.FaceMesh network to retrieve face landmark coordinates
            - This stage of the pipeline can optionally be configured to also do either or both of the following:
                - Set all non-face pixels (i.e. pixels outside the mesh formed by the face landmarks) to 0
                - Calculate accurate roll angle using eye landmark coordinates of mp.solutions.face_mesh.FaceMesh (like in step 2.1) and correct the roll of the
                  face by rotating the image (and landmarks) in the opposite direction around the midpoint between the eyes (enabled by default)
            - For more about this network, visit https://solutions.mediapipe.dev/face_mesh
        4. The image is then cropped to the minimum rectangle spanning all face landmarks

    - The eye coordinates from the mp.solutions.face_detection.FaceDetection network in step 1 are not used to correct the roll as they are not
      accurate enough and are only good enough for getting an approximate roll angle

    - mp.solutions.face_mesh.FaceMesh has an integrated mp.solutions.face_detection.FaceDetection model. However, since the integrated network is configured to only
      run the short-range model (see https://solutions.mediapipe.dev/face_detection#model_selection), it can't detect faces further than 2 metres away. Having a
      separate mp.solutions.face_detection.FaceDetection model makes the pipeline more configurable to suit requirements

    - mp.solutions.face_mesh.FaceMesh requires a maximum number of faces (max_num_faces) to be specified during initialisation, while mp.solutions.face_detection.FaceDetection
      does not have such limitation. Hence, Step 1 uses mp.solutions.face_detection.FaceDetection to get all potential faces, which are then verified with mp.solutions.face_mesh.FaceMesh
      (max_num_faces=1). I.e., if mp.solutions.face_mesh.FaceMesh can detect landmarks in a potential face detected by mp.solutions.face_detection.FaceDetection, it is taken further in the pipeline
    """

    # face_detector_model_selection values
    SHORT_RANGE = 0
    LONG_RANGE = 1

    # landmark_detector_static_image_mode values
    STATIC_MODE = True
    TRACKING_MODE = False


    def __init__(self, min_face_detector_confidence=0.5, face_detector_model_selection=LONG_RANGE,
                 landmark_detector_static_image_mode=STATIC_MODE, min_landmark_detector_confidence=0.5):
        """
        Initialise a FaceCropper object.
        :param min_face_detector_confidence:
        From mp.solutions.face_detection.FaceDetection documentation:
        "min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.
        See details in https://solutions.mediapipe.dev/face_detection#min_detection_confidence". Defaults to 0.5.
        :param face_detector_model_selection:
        From mp.solutions.face_detection.FaceDetection documentation:
        "0 (FaceCropper.SHORT_RANGE) or 1 (FaceCropper.LONG_RANGE). 0 to select a short-range model
        that works best for faces within 2 meters from the camera, and 1 for a full-range model best for faces within 5 meters.
        See details in https://solutions.mediapipe.dev/face_detection#model_selection".
        1 works well as a general purpose model that detects both close and long range faces, whereas 0 is better for detecting
        close range faces with higher yaw, pitch, or 90+ degree roll. Defaults to 1.
        :param landmark_detector_static_image_mode:
        From mp.solutions.face_mesh.FaceMesh documentation:
        "Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream. See details in
        https://solutions.mediapipe.dev/face_mesh#static_image_mode". Should only be set to False (FaceCropper.TRACKING_MODE) if the images passed to this
        pipeline are from the same sequence, AND there is always the same one face in the sequence. Defaults to True (FaceCropper.STATIC_MODE).
        :param min_landmark_detector_confidence:
        From mp.solutions.face_mesh.FaceMesh documentation:
        "Minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful. See details in
        https://google.github.io/mediapipe/solutions/face_mesh#min_detection_confidence". Defaults to 0.5.
        """

        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_face_detector_confidence,
                                                                       model_selection=face_detector_model_selection)

        self.landmark_detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                                 static_image_mode=landmark_detector_static_image_mode,
                                                                 min_detection_confidence=min_landmark_detector_confidence)


    def get_faces(self, image, remove_background=False, correct_roll=True):
        """
        Crop out (and optionally correct the roll and/or remove background of) each detected face in the specified image and return them in a list.
        :param image: A numpy.ndarray RGB image containing faces to be cropped
        :param remove_background: Whether non-face (i.e. background) pixels should be set to 0. Defaults to False
        :param correct_roll: Whether the roll in faces should be corrected. Defaults to True
        :return: A list of numpy.ndarray RGB images containing the cropped faces
        """

        face_images = []

        detected_faces = self.face_detector.process(image).detections
        if detected_faces is None: return face_images

        for face in detected_faces:
            # The mp.solutions.face_detection.FaceDetection network may rarely 'find' a face completely outside the image, so ignore those
            if 0 <= face.location_data.relative_bounding_box.xmin <= 1 and 0 <= face.location_data.relative_bounding_box.ymin <= 1:
                inflation_factor = _get_bounding_box_inflation_factor(face.location_data.relative_keypoints[:2])
                inflated_face_image = _get_inflated_face_image(image, face.location_data.relative_bounding_box, inflation_factor)
                detected_landmarks = self.landmark_detector.process(inflated_face_image).multi_face_landmarks

                if detected_landmarks is not None:
                    face_landmarks = detected_landmarks[0].landmark

                    if remove_background:
                        inflated_face_image = _get_segmented_face_image(inflated_face_image, _FACE_MESH, face_landmarks)

                    if correct_roll:
                        inflated_face_image, face_landmarks = _get_roll_corrected_image_and_landmarks(inflated_face_image, face_landmarks)
                    else:
                        face_landmarks = np.ndarray.astype(np.rint(np.array([
                            np.multiply([landmark.x for landmark in face_landmarks], inflated_face_image.shape[1]),
                            np.multiply([landmark.y for landmark in face_landmarks], inflated_face_image.shape[0])])), np.int64)

                    face_images.append(
                        _crop_within_bounds(
                            inflated_face_image,
                            face_landmarks[1, np.argmin(face_landmarks[1, :])], face_landmarks[1, np.argmax(face_landmarks[1, :])],
                            face_landmarks[0, np.argmin(face_landmarks[0, :])], face_landmarks[0, np.argmax(face_landmarks[0, :])]
                        )
                    )

        return face_images


    def get_faces_debug(self, image, remove_background=False, correct_roll=True):
        """
        Identical to get_faces(), except it displays the following debug information:
            - Annotations of face detection boxes and eye coordinates from mp.solutions.face_detection.FaceDetection, approximate roll angle,
              inflation factor, and inflated face detection boxes
            - Annotations of landmarks detected from mp.solutions.face_mesh.FaceMesh, eye coordinates, and roll angle
            - Output images for each (remove_background, correct_roll) combination
        :param image: A numpy.ndarray RGB image containing faces to be cropped
        :param remove_background: Whether non-face (i.e. background) pixels should be set to 0
        :param correct_roll: Whether the roll in faces should be corrected
        :return: A list of numpy.ndarray RGB images containing the cropped faces
        """

        face_images = []

        detected_faces = self.face_detector.process(image).detections
        if detected_faces is None: return face_images

        # IMAGE_DEBUG START #
        image_debug = image.copy()
        for face in detected_faces:
            # Face detection box
            face_box = face.location_data.relative_bounding_box
            cv2.rectangle(
                image_debug,
                (round(face_box.xmin * image.shape[1]), round(face_box.ymin * image.shape[0])),
                (round((face_box.xmin + face_box.width) * image.shape[1]),
                 round((face_box.ymin + face_box.height) * image.shape[0])),
                (0, 255, 0)
            )
            # Eye coordinates
            eye_coordinates = face.location_data.relative_keypoints[:2]
            inflation_factor = _get_bounding_box_inflation_factor(eye_coordinates)
            for eye_coordinate in eye_coordinates:
                cv2.circle(image_debug, (round(eye_coordinate.x * image.shape[1]), round(eye_coordinate.y * image.shape[0])), 1, (0, 255, 0))
            # Roll line
            cv2.line(
                image_debug,
                (round(eye_coordinates[0].x * image.shape[1]), round(eye_coordinates[0].y * image.shape[0])),
                (round(eye_coordinates[1].x * image.shape[1]), round(eye_coordinates[1].y * image.shape[0])),
                (255, 0, 0)
            )
            # Horizontal line
            cv2.line(
                image_debug,
                (round(eye_coordinates[0].x * image.shape[1]), round(eye_coordinates[0].y * image.shape[0])),
                (round(eye_coordinates[1].x * image.shape[1]), round(eye_coordinates[0].y * image.shape[0])),
                (255, 0, 0)
            )
            # Roll angle
            cv2.putText(
                image_debug,
                'roll_angle: {:.2f}'.format(_get_face_roll_angle([eye_coordinates[1].x, 1 - eye_coordinates[1].y], [eye_coordinates[0].x, 1 - eye_coordinates[0].y])),
                (round(eye_coordinates[0].x * image.shape[1]), round(eye_coordinates[0].y * image.shape[0] + 40)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255)
            )
            # Inflation factor
            cv2.putText(
                image_debug,
                'inflation_factor: {:.2f}'.format(inflation_factor),
                (round(eye_coordinates[0].x * image.shape[1]), round(eye_coordinates[0].y * image.shape[0] + 80)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255)
            )
            # Inflated face detection box
            width_inflation, height_inflation = face_box.width * inflation_factor, face_box.height * inflation_factor
            cv2.rectangle(
                image_debug,
                (round((face_box.xmin - width_inflation / 2) * image.shape[1]), round((face_box.ymin + face_box.height + height_inflation / 2) * image.shape[0])),
                (round((face_box.xmin + face_box.width + width_inflation / 2) * image.shape[1]), round((face_box.ymin - height_inflation / 2) * image.shape[0])),
                (0, 0, 255)
            )
        cv2.imshow('image_debug', cv2.cvtColor(image_debug, cv2.COLOR_RGB2BGR))
        # IMAGE_DEBUG END #

        for face in detected_faces:
            # The mp.solutions.face_detection.FaceDetection network may rarely 'find' a face completely outside the image, so ignore those
            if 0 <= face.location_data.relative_bounding_box.xmin <= 1 and 0 <= face.location_data.relative_bounding_box.ymin <= 1:
                inflation_factor = _get_bounding_box_inflation_factor(face.location_data.relative_keypoints[:2])
                inflated_face_image = _get_inflated_face_image(image, face.location_data.relative_bounding_box, inflation_factor)
                detected_landmarks = self.landmark_detector.process(inflated_face_image).multi_face_landmarks

                if detected_landmarks is not None:
                    face_landmarks = detected_landmarks[0].landmark

                    # INFLATED_FACE_IMAGE_DEBUG START #
                    inflated_face_image_debug = inflated_face_image.copy()
                    for i, landmark in enumerate(face_landmarks):
                        # Eye landmarks
                        if i in _LEFT_EYE_LANDMARK_INDICES or i in _RIGHT_EYE_LANDMARK_INDICES:
                            cv2.circle(inflated_face_image_debug, (round(landmark.x * inflated_face_image.shape[1]), round(landmark.y * inflated_face_image.shape[0])), 1, (255, 0, 0))
                        # Face landmarks
                        else:
                            cv2.circle(inflated_face_image_debug, (round(landmark.x*inflated_face_image.shape[1]), round(landmark.y*inflated_face_image.shape[0])), 1, (0, 255, 0))
                    # Eye coordinates
                    left_eye_centre, right_eye_centre = _get_left_and_right_eye_centres(
                        [face_landmarks[landmark] for landmark in _LEFT_EYE_LANDMARK_INDICES],
                        [face_landmarks[landmark] for landmark in _RIGHT_EYE_LANDMARK_INDICES])
                    for eye_coordinate in [left_eye_centre, right_eye_centre]:
                        cv2.circle(inflated_face_image_debug, (round(eye_coordinate[0]*inflated_face_image.shape[1]), round((1-eye_coordinate[1])*inflated_face_image.shape[0])), 2, (255, 255, 0))
                    # Eye midpoint
                    eyes_midpoint = _get_eyes_midpoint(left_eye_centre, right_eye_centre, inflated_face_image.shape)
                    cv2.circle(inflated_face_image_debug, (eyes_midpoint[0], inflated_face_image.shape[0] - eyes_midpoint[1]), 2, (0, 0, 255))
                    # Roll line
                    cv2.line(
                        inflated_face_image_debug,
                        (round(left_eye_centre[0] * inflated_face_image.shape[1]), round((1-left_eye_centre[1]) * inflated_face_image.shape[0])),
                        (round(right_eye_centre[0] * inflated_face_image.shape[1]), round((1-right_eye_centre[1]) * inflated_face_image.shape[0])),
                        (255, 0, 0)
                    )
                    # Horizontal line
                    cv2.line(
                        inflated_face_image_debug,
                        (round(right_eye_centre[0] * inflated_face_image.shape[1]), round((1-right_eye_centre[1]) * inflated_face_image.shape[0])),
                        (round(left_eye_centre[0] * inflated_face_image.shape[1]), round((1-right_eye_centre[1]) * inflated_face_image.shape[0])),
                        (255, 0, 0)
                    )
                    # Roll angle
                    cv2.putText(
                        inflated_face_image_debug,
                        'roll_angle: {:.2f}'.format(_get_face_roll_angle(left_eye_centre, right_eye_centre)),
                        (round(right_eye_centre[0] * inflated_face_image.shape[1]), round((1 - right_eye_centre[1]) * inflated_face_image.shape[0] + 20)),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255)
                    )
                    inflated_face_image_segmented = _get_segmented_face_image(inflated_face_image, _FACE_MESH, face_landmarks)
                    cv2.imshow('inflated_face_image_debug', cv2.cvtColor(np.row_stack((
                                np.column_stack((
                                        inflated_face_image_debug,
                                        inflated_face_image_segmented
                                    )
                                ),
                                np.column_stack((
                                        _get_roll_corrected_image_and_landmarks(inflated_face_image, face_landmarks)[0],
                                        _get_roll_corrected_image_and_landmarks(inflated_face_image_segmented, face_landmarks)[0]
                                    )
                                )
                            )
                        ), cv2.COLOR_RGB2BGR)
                    )
                    # INFLATED_FACE_IMAGE_DEBUG END #

                    # OUTPUT_IMAGE_DEBUG START #
                    landmarks = np.ndarray.astype(np.rint(np.array([
                            np.multiply([landmark.x for landmark in face_landmarks], inflated_face_image.shape[1]),
                            np.multiply([landmark.y for landmark in face_landmarks], inflated_face_image.shape[0])])), np.int64)
                    cv2.imshow('output_image_debug', cv2.cvtColor(np.column_stack((
                                   _crop_within_bounds(
                                        inflated_face_image,
                                        landmarks[1, np.argmin(landmarks[1, :])],
                                        landmarks[1, np.argmax(landmarks[1, :])],
                                        landmarks[0, np.argmin(landmarks[0, :])],
                                        landmarks[0, np.argmax(landmarks[0, :])]
                                    ),
                                   _crop_within_bounds(
                                       inflated_face_image_segmented,
                                       landmarks[1, np.argmin(landmarks[1, :])],
                                       landmarks[1, np.argmax(landmarks[1, :])],
                                       landmarks[0, np.argmin(landmarks[0, :])],
                                       landmarks[0, np.argmax(landmarks[0, :])]
                                   ),
                            )
                        ), cv2.COLOR_RGB2BGR)
                    )
                    # OUTPUT_IMAGE_DEBUG END #

                    # OUTPUT_IMAGE_CORRECTED_DEBUG START #
                    corrected_inflated_face_image, corrected_landmarks = _get_roll_corrected_image_and_landmarks(inflated_face_image, face_landmarks)
                    corrected_inflated_face_image_segmented, corrected_landmarks = _get_roll_corrected_image_and_landmarks(inflated_face_image_segmented, face_landmarks)
                    cv2.imshow('output_image_corrected_debug', cv2.cvtColor(np.column_stack((
                                   _crop_within_bounds(
                                        corrected_inflated_face_image,
                                        corrected_landmarks[1, np.argmin(corrected_landmarks[1, :])],
                                        corrected_landmarks[1, np.argmax(corrected_landmarks[1, :])],
                                        corrected_landmarks[0, np.argmin(corrected_landmarks[0, :])],
                                        corrected_landmarks[0, np.argmax(corrected_landmarks[0, :])]
                                    ),
                                   _crop_within_bounds(
                                       corrected_inflated_face_image_segmented,
                                       corrected_landmarks[1, np.argmin(corrected_landmarks[1, :])],
                                       corrected_landmarks[1, np.argmax(corrected_landmarks[1, :])],
                                       corrected_landmarks[0, np.argmin(corrected_landmarks[0, :])],
                                       corrected_landmarks[0, np.argmax(corrected_landmarks[0, :])]
                                   ),
                            )
                        ), cv2.COLOR_RGB2BGR)
                    )
                    # OUTPUT_IMAGE_CORRECTED_DEBUG END #

                    if remove_background:
                        inflated_face_image = _get_segmented_face_image(inflated_face_image, _FACE_MESH, face_landmarks)

                    if correct_roll:
                        inflated_face_image, face_landmarks = _get_roll_corrected_image_and_landmarks(inflated_face_image, face_landmarks)
                    else:
                        face_landmarks = np.ndarray.astype(np.rint(np.array([
                            np.multiply([landmark.x for landmark in face_landmarks], inflated_face_image.shape[1]),
                            np.multiply([landmark.y for landmark in face_landmarks], inflated_face_image.shape[0])])), np.int64)

                    face_images.append(
                        _crop_within_bounds(
                            inflated_face_image,
                            face_landmarks[1, np.argmin(face_landmarks[1, :])], face_landmarks[1, np.argmax(face_landmarks[1, :])],
                            face_landmarks[0, np.argmin(face_landmarks[0, :])], face_landmarks[0, np.argmax(face_landmarks[0, :])]
                        )
                    )

        return face_images
