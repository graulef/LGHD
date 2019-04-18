/**
 * @file   image_io.h
 * @author Carlos H. Villa Pinto (chvillap@gmail.com), modified by Felix Graule (felix.graule@gmail.com)
 *
 * @copyright Copyright (c) 2016 Carlos Henrique Villa Pinto
 * @license GPL v2.0
 */

#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <itkImage.h>
#include <itkVector.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "types.h"


/**
 * @fn read_image
 *
 * @brief Reads an image from file using ITK.
 *
 * @param[in] filename Image file name.
 *
 * @returns A smart pointer to the image object in ITK's format.
 */
template <class TPixel, size_t dims>
typename itk::Image<TPixel, dims>::Pointer read_image(std::string filename){
    // Filename can't be empty.
    assert(!filename.empty());

    typedef itk::Image<TPixel, dims>     TImage;
    typedef itk::ImageFileReader<TImage> TReader;

    // Initialize the reader.
    typename TReader::Pointer reader = TReader::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    // Reads the file.
    return reader->GetOutput();
}


/**
 * @fn write_image
 *
 * @brief Writes an image using ITK.
 *
 * @param[in] filename Image file name.
 * @param[in] image    Image object in ITK's format.
 */
template <class TPixel, size_t dims>
void write_image(std::string filename,
                 typename itk::Image<TPixel, dims>::Pointer image){
    // Filename can't be empty and image can't be null.
    assert(!filename.empty());
    assert(image.IsNotNull());

    typedef itk::Image<TPixel, dims>     TImage;
    typedef itk::ImageFileWriter<TImage> TWriter;

    // Initialize the writer.
    typename TWriter::Pointer writer = TWriter::New();
    writer->SetInput(image);
    writer->SetFileName(filename.c_str());

    // Writes the file.
    writer->Update();
}


/**
 * @fn array2image
 *
 * @brief Creates an ITK image from some array of pixel data.
 *
 * @param[in] data      Pixel data array.
 * @param[in] sizes     Number of array elements per dimension.
 * @param[in] reference Optional reference for 3D images.
 *
 * @returns A smart pointer to the image object in ITK's format.
 */
template <class TPixel, size_t dims>
typename itk::Image<TPixel, dims>::Pointer array2image(
    TPixel         *data,
    triple<size_t>  sizes,
    typename itk::Image<TPixel, dims>::Pointer reference = NULL){
    // Data array can't be null.
    assert(data != NULL);

    typedef itk::Image<TPixel, dims> TImage;

    // Set the image sizes in ITK format.
    typename TImage::SizeType sizes2;
    for (size_t d = 0; d < dims; ++d)
        sizes2[d] = sizes[d];

    // Set the image index.
    typename TImage::IndexType start;
    for (size_t d = 0; d < dims; ++d)
        start[d] = 0;

    // Set the image region.
    typename TImage::RegionType region;
    region.SetSize(sizes2);
    region.SetIndex(start);

    typename TImage::Pointer image = TImage::New();
    image->SetRegions(region);

    // Origin, spacing and direction can be taken from the reference image.
    if (reference.IsNotNull()) {
        image->SetOrigin(reference->GetOrigin());
        image->SetSpacing(reference->GetSpacing());
        image->SetDirection(reference->GetDirection());
    }

    // Initialize the pixel data.
    image->Allocate();
    image->FillBuffer(TPixel());

    // Set the image pixels.
    itk::ImageRegionIterator<TImage> iterator(image, image->GetBufferedRegion());
    size_t i = 0;
    while (!iterator.IsAtEnd()) {
        iterator.Set(data[i]);
        ++iterator;
        ++i;
    }

    return image;
}


/**
 * @fn image2array
 *
 * @brief Gets an array of pixel data from some ITK image.
 *
 * @param[in] image Image object in ITK's format.
 *
 * @returns A flattened array of pixel data.
 */
template <class TPixel, size_t dims>
TPixel* image2array(typename itk::Image<TPixel, dims>::Pointer image){
    // Image can't be null.
    assert(image.IsNotNull());

    typedef itk::Image<TPixel, dims> TImage;

    // Get the image size.
    typename TImage::SizeType sizes = image->GetBufferedRegion().GetSize();
    size_t total_size = 1;
    for (size_t d = 0; d < dims; ++d)
        total_size *= sizes[d];

    // Allocate a flattened array of pixel data.
    TPixel *data = new TPixel[total_size]();

    // Get the image pixels.
    itk::ImageRegionConstIterator<TImage> iterator(image, image->GetBufferedRegion());
    size_t i = 0;
    while (!iterator.IsAtEnd()) {
        data[i] = iterator.Get();
        ++iterator;
        ++i;
    }

    return data;
}

#endif
