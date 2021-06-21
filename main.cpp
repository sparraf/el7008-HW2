#define _DEBUG

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

using namespace std;
using namespace cv;

Mat harrisFilter(Mat input)
{
    Mat input_gray;
    Mat harris;
    harris = Mat::zeros(input.rows, input.cols, CV_32FC1);

    // Paso 1: Suavizar y obtener imagen en escala de grises
    cvtColor(input, input_gray, COLOR_BGR2GRAY);
    GaussianBlur(input_gray, input_gray, Size(3,3), 2, 2);

    // Paso 2: Calcular derivadas en x e y
    Mat ix, iy;
    Scharr(input_gray, ix, CV_32FC1, 1, 0);
    Scharr(input_gray, iy, CV_32FC1, 0, 1);

    // Paso 3: Calcular ixx, ixy, iyy
    Mat ixx, ixy, iyy;
    ixx = ix.mul(ix);
    ixy = ix.mul(iy);
    iyy = iy.mul(iy);

    // Paso 4: Suavizar momentos ixx, ixy, iyy
    GaussianBlur(ixx, ixx, Size(3,3), 2.0/0.7, 2.0/0.7);
    GaussianBlur(ixy, ixy, Size(3,3), 2.0/0.7, 2.0/0.7);
    GaussianBlur(iyy, iyy, Size(3,3), 2.0/0.7, 2.0/0.7);

    // Paso 5: Calcular cornerness
    Mat det, Tr;
    det = ixx.mul(iyy) - ixy.mul(ixy);
    Tr = (ixx + iyy);

    harris = det - 0.04*(Tr.mul(Tr));

    // Paso 6: Transformar la imagen para que quede con valores en el rango 0-255
    Mat output;
    normalize(harris, harris, 0, 255, NORM_MINMAX, CV_32FC1);
    convertScaleAbs(harris, output);
    return output;
}

vector<KeyPoint> getHarrisPoints(Mat harris, int val)
{
    vector<KeyPoint> points;
    double maxVal;
    Point maxLoc;
    // Recorrer la imagen con una ventana de 3x3, y agregar el punto medio de la ventana
    // al vector points si es un máximo local y es mayor a val.
    for (int r=0; r<harris.rows-2; r++)
    {
        for (int c=0; c<harris.cols-2; c++)
        {
            // Paso 1: Obtener maximo local
            minMaxLoc(harris(Rect(c, r, 3, 3)), nullptr, &maxVal, nullptr, &maxLoc);
            // Paso 2: Verificar que se encuentre en el centro de la imagen
            if (maxLoc.x == 1 && maxLoc.y == 1)
            {
                // Paso 3: Verificar que el punto tenga un valor mayor a val
                if (maxVal > val)
                {
                    // Paso 4: Agregar al vector points y dibujar un circulo que indicque la ubicacion
                    // del punto en la imagen
                    maxLoc.x += c;
                    maxLoc.y += r;
                    points.emplace_back(KeyPoint(maxLoc, 1));
                }
            }
        }
    }
    return points;
}

int main(void)
{
    Mat imleft = imread("../image_pairs/left5.jpg");
    Mat imright = imread("../image_pairs/right5.jpg")  ;

    if(imleft.empty() || imright.empty()) // No encontro la imagen
    {
        cout << "Imagen no encontrada" << endl;
        return 1;
    }

    // Crear matrices donde se guardaran las imagenes con los puntos de interes
    Mat impointsleft, impointsright;
    impointsleft = imleft.clone();
    impointsright = imright.clone();

    // Crear matrices de harris
    Mat harrisleft, harrisright;
    harrisleft = harrisFilter(imleft);
    harrisright = harrisFilter(imright);
    // Guardar las imagenes de harris
    imwrite("../harrisleft.jpg", harrisleft); // Grabar imagen
    imwrite("../harrisright.jpg", harrisright); // Grabar imagen

    // Obtener vectores de puntos de interes
    // Por defecto se recomienda dejar un umbral de 115, pero a veces puede ser necesario bajarlo
    vector<KeyPoint> pointsleft = getHarrisPoints(harrisleft , 115);
    vector<KeyPoint> pointsright = getHarrisPoints(harrisright, 115);

    // Dibujar los puntos de interes
    drawKeypoints(impointsleft, pointsleft, impointsleft);
    drawKeypoints(impointsright, pointsright, impointsright);
    // Guardar imagen con los puntos de interes dibujados
    imwrite("../impointsleft.jpg", impointsleft); // Grabar imagen
    imwrite("../impointsright.jpg", impointsright); // Grabar imagen

    // Crear descriptores ORB
    Ptr<ORB> orb = ORB::create();
    Mat descrleft, descrright;
    orb->compute(imleft, pointsleft, descrleft);
    orb->compute(imright, pointsright, descrright);

    // Hacer matching
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descrleft, descrright, matches);

    // Dibujar matches y guardarlos
    Mat img_matches;
    drawMatches(imleft, pointsleft, imright, pointsright, matches, img_matches);
    imwrite("../img_matches.jpg", img_matches);

    // Guardar los puntos de interes usados para matching
    vector<Point2f> points1, points2;
    for (int i=0; i<matches.size(); i++)
    {
        points1.push_back(pointsleft[matches[i].queryIdx].pt);
        points2.push_back(pointsright[matches[i].trainIdx].pt);
    }

    // Encontrar transformacion que relaciona las dos imagenes
    Mat homography;
    homography = findHomography(points2, points1, RANSAC);

    /*
    double ty = homography.at<double>(1,2);
    homography.at<double>(1,2)-=ty;
    */

    // Proyectar imagen de la derecha en imwarp
    Mat imwarp;
    warpPerspective(imright, imwarp, homography, Size(imleft.cols * 2, imleft.rows * 1.8));

    // Alinear las imagenes
    Mat imFused(imwarp.size(), CV_8UC3);
    // Primero dibujar la imagen de la derecha proyectada, luego la de la izquierda
    imwarp.copyTo(imFused(Rect(0, 0, imwarp.cols, imwarp.rows)));
    imleft.copyTo(imFused(Rect(0, 0, imleft.cols, imleft.rows)));
    //imleft.copyTo(imFused(Rect(0, 340, imleft.cols, imleft.rows)));
    // Guardar la imagen fusionada
    imwrite("../imfused.jpg", imFused);

    return 0; // Sale del programa normalmente

}

