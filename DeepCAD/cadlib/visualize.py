import os
import random
from copy import copy

import trimesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Lin, gp_Pln, gp_Pnt, gp_Vec
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Shape, ShapeFix_Wire
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Extend.DataExchange import write_stl_file
from trimesh.sample import sample_surface

from .curves import *
from .extrude import *
from .sketch import Loop, Profile


def vec2CADsolid(vec, is_numerical=True, n=256):
    cad = CADSequence.from_vector(vec, is_numerical=is_numerical, n=256)
    cad = create_CAD(cad)
    return cad


def create_CAD(cad_seq: CADSequence):
    """create a 3D CAD model from CADSequence. Only support extrude with boolean operation."""
    # print("cad-seq:")
    # print(cad_seq)
    body = create_by_extrude(cad_seq.seq[0])
    for extrude_op in cad_seq.seq[1:]:
        new_body = create_by_extrude(extrude_op)
        if extrude_op.operation == EXTRUDE_OPERATIONS.index(
            "NewBodyFeatureOperation"
        ) or extrude_op.operation == EXTRUDE_OPERATIONS.index("JoinFeatureOperation"):
            body = BRepAlgoAPI_Fuse(body, new_body).Shape()
        elif extrude_op.operation == EXTRUDE_OPERATIONS.index("CutFeatureOperation"):
            body = BRepAlgoAPI_Cut(body, new_body).Shape()
        elif extrude_op.operation == EXTRUDE_OPERATIONS.index(
            "IntersectFeatureOperation"
        ):
            body = BRepAlgoAPI_Common(body, new_body).Shape()
    return body


def create_by_extrude(extrude_op: Extrude):
    """create a solid body from Extrude instance."""
    profile = copy(
        extrude_op.profile
    )  # use copy to prevent changing extrude_op internally
    profile.denormalize(extrude_op.sketch_size)

    sketch_plane = copy(extrude_op.sketch_plane)
    sketch_plane.origin = extrude_op.sketch_pos

    face = create_profile_face(profile, sketch_plane)

    normal = gp_Dir(*extrude_op.sketch_plane.normal)
    ext_vec = gp_Vec(normal).Multiplied(extrude_op.extent_one)
    body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
    if extrude_op.extent_type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
        body_sym = BRepPrimAPI_MakePrism(face, ext_vec.Reversed()).Shape()
        body = BRepAlgoAPI_Fuse(body, body_sym).Shape()
    if extrude_op.extent_type == EXTENT_TYPE.index("TwoSidesFeatureExtentType"):
        ext_vec = gp_Vec(normal.Reversed()).Multiplied(extrude_op.extent_two)
        body_two = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
        body = BRepAlgoAPI_Fuse(body, body_two).Shape()
    return body


def create_profile_face(profile: Profile, sketch_plane: CoordSystem):
    """create a face from a sketch profile and the sketch plane"""
    origin = gp_Pnt(*sketch_plane.origin)
    normal = gp_Dir(*sketch_plane.normal)
    x_axis = gp_Dir(*sketch_plane.x_axis)
    gp_face = gp_Pln(gp_Ax3(origin, normal, x_axis))

    all_loops = [create_loop_3d(loop, sketch_plane) for loop in profile.children]
    if all_loops and not all_loops[0].IsNull():
        topo_face = BRepBuilderAPI_MakeFace(gp_face, all_loops[0])
        for loop in all_loops[1:]:
            if not loop.IsNull():
                topo_face.Add(loop.Reversed())
        topo_face = topo_face.Face()

        fix_face = ShapeFix_Face(topo_face)
        fix_face.Perform()
        # fix_face.FixMissingSeam()
        fix_face.FixAddNaturalBound()
        fix_face.FixOrientation()
        fix_face.FixIntersectingWires()
        fix_face.FixPeriodicDegenerated()

        return fix_face.Face()
    return None


# def create_loop_3d(loop: Loop, sketch_plane: CoordSystem):
#     """create a 3D sketch loop"""
#     topo_wire = BRepBuilderAPI_MakeWire()
#     for curve in loop.children:
#         topo_edge = create_edge_3d(curve, sketch_plane)
#         if topo_edge == -1: # omitted
#             continue
#         topo_wire.Add(topo_edge)
#     return topo_wire.Wire()


def create_loop_3d(loop: Loop, sketch_plane: CoordSystem):
    """create a 3D sketch loop"""
    topo_wire = ShapeExtend_WireData()
    for curve in loop.children:
        topo_edge = create_edge_3d(curve, sketch_plane)
        if topo_edge == -1:  # omitted
            continue
        topo_wire.Add(topo_edge)

    fix_wire = ShapeFix_Wire()
    fix_wire.Load(topo_wire)

    fix_wire.FixReorder()
    fix_wire.FixConnected()
    fix_wire.FixClosed()
    fix_wire.FixGaps2d()
    fix_wire.FixGaps3d()
    fix_wire.FixEdgeCurves()
    fix_wire.FixDegenerated()
    fix_wire.FixSelfIntersection()
    fix_wire.FixLacking()
    fix_wire.FixNotchedEdges()
    fix_wire.FixConnected()

    fix_wire.Perform()
    # print(fix_wire.NbEdges())

    return fix_wire.WireAPIMake()


def create_edge_3d(curve: CurveBase, sketch_plane: CoordSystem):
    """create a 3D edge"""
    if isinstance(curve, Line):
        if np.allclose(curve.start_point, curve.end_point):
            return -1
        start_point = point_local2global(curve.start_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        topo_edge = BRepBuilderAPI_MakeEdge(start_point, end_point)
    elif isinstance(curve, Circle):
        center = point_local2global(curve.center, sketch_plane)
        axis = gp_Dir(*sketch_plane.normal)
        gp_circle = gp_Circ(gp_Ax2(center, axis), abs(float(curve.radius)))
        topo_edge = BRepBuilderAPI_MakeEdge(gp_circle)
    elif isinstance(curve, Arc):
        start_point = point_local2global(curve.start_point, sketch_plane)
        mid_point = point_local2global(curve.mid_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        arc = GC_MakeArcOfCircle(start_point, mid_point, end_point).Value()
        topo_edge = BRepBuilderAPI_MakeEdge(arc)
    else:
        raise NotImplementedError(type(curve))

    return topo_edge.Edge()


def point_local2global(point, sketch_plane: CoordSystem, to_gp_Pnt=True):
    """convert point in sketch plane local coordinates to global coordinates"""
    g_point = (
        point[0] * sketch_plane.x_axis
        + point[1] * sketch_plane.y_axis
        + sketch_plane.origin
    )
    if to_gp_Pnt:
        return gp_Pnt(*g_point)
    return g_point


def CADsolid2pc(shape, n_points, name=None):
    """convert opencascade solid to point clouds"""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    if bbox.IsVoid():
        raise ValueError("box check failed")
    if name is None:
        name = random.randint(100000, 999999)
    tmp_file_dir = os.path.join("data", "tmp_stl")
    if not os.path.isdir(tmp_file_dir):
        os.makedirs(tmp_file_dir)
    tmp_file_path = os.path.join(
        tmp_file_dir, "tmp_out_{}.stl".format(name.split("/")[-1])
    )
    write_stl_file(shape, tmp_file_path)
    out_mesh = trimesh.load(tmp_file_path)
    os.system("rm {}".format(tmp_file_path))
    out_pc, _ = sample_surface(out_mesh, n_points)
    return out_pc
