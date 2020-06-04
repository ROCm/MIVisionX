/*
 * Copyright (c) 2012-2014 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#include "test.h"
#include "test_bmp.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct GrfmtReader;
typedef unsigned char uchar;

typedef void (*GrfmtReleaseReader)(struct GrfmtReader* reader);
typedef struct GrfmtReader* (*GrfmtReadHeader)(const uchar* data, int size, int* width, int* height, int* iscolor);
typedef int (*GrfmtReadData)(struct GrfmtReader* reader, uchar* data, int step, int color);

typedef struct GrfmtReader
{
    GrfmtReleaseReader release;
    GrfmtReadData read;
}
GrfmtReader;

typedef struct PaletteEntry
{
    uchar b, g, r, a;
}
PaletteEntry;

typedef enum BmpCompression
{
    BMP_RGB = 0,
    BMP_RLE8 = 1,
    BMP_RLE4 = 2,
    BMP_BITFIELDS = 3
}
BmpCompression;

typedef struct GrfmtBmpReader
{
    GrfmtReader base;
    PaletteEntry m_palette[256];
    const uchar* m_data;
    int m_datasize;
    int m_offset;
    int m_width;
    int m_height;
    int m_bpp;
    int m_origin;
    BmpCompression m_rle_code;
}
GrfmtBmpReader;


static void releaseBmpReader(struct GrfmtReader* reader)
{
    if(reader)
        free(reader);
}

#define GET_DWORD(p) ((p) += 4, ((p)[-4] | ((p)[-3]<<8) | ((p)[-2]<<16) | ((p)[-1]<<24)))

static int isColorPalette(const PaletteEntry* pal, int bpp)
{
    int j, clrused = 1 << bpp;
    for( j = 0; j < clrused; j++ )
        if( pal[j].b != pal[j].g || pal[j].b != pal[j].r )
            return 1;
    return 0;
}


#define  SCALE  14
#define  cR  (int)(0.299*(1 << SCALE) + 0.5)
#define  cG  (int)(0.587*(1 << SCALE) + 0.5)
#define  cB  ((1 << SCALE) - cR - cG)

static void cvtRGBToGray( const uchar* src, uchar* gray, int n, int scn, int blue_idx )
{
    int i;
    for( i = 0; i < n; i++, src += scn )
    {
        gray[i] = (uchar)((src[blue_idx]*cB + src[1]*cG + src[blue_idx^2]*cR + (1 << (SCALE-1))) >> SCALE);
    }
}

static void cvtRGBToRGB( const uchar* src, uchar* dst, int n,
                         int scn, int sblue_idx, int dcn, int dblue_idx )
{
    int i;
    for( i = 0; i < n; i++, src += scn, dst += dcn )
    {
        dst[dblue_idx] = src[sblue_idx];
        dst[1] = src[1];
        dst[dblue_idx^2] = src[sblue_idx^2];
        if( dcn == 4 )
            dst[3] = scn < 4 ? 255 : src[3];
    }
}

static void fillColorRow8( uchar* data, const uchar* indices, int n,
                           const PaletteEntry* palette, int dcn, int dblue_idx )
{
    int i;
    for( i = 0; i < n; i++, data += dcn )
    {
        const PaletteEntry* p = palette + indices[i];
        data[dblue_idx] = p->b;
        data[1] = p->g;
        data[dblue_idx^2] = p->r;
        if( dcn == 4 )
            data[3] = 255;
    }
}


static void fillGrayRow8( uchar* data, const uchar* indices, int n, const uchar* palette )
{
    int i;
    for( i = 0; i < n; i++ )
    {
        data[i] = palette[indices[i]];
    }
}


static void fillColorRow4( uchar* data, const uchar* indices, int n,
                           const PaletteEntry* palette, int dcn, int dblue_idx )
{
    int i = 0;
    for( ; i <= n - 2; i += 2, data += dcn*2 )
    {
        int idx = *indices++;
        const PaletteEntry* p0 = palette + (idx >> 4);
        const PaletteEntry* p1 = palette + (idx & 15);
        data[dblue_idx] = p0->b;
        data[1] = p0->g;
        data[dblue_idx^2] = p0->r;
        data[dcn + dblue_idx] = p1->b;
        data[dcn + 1] = p1->g;
        data[dcn + (dblue_idx^2)] = p1->r;
        if( dcn == 4 )
            data[3] = data[7] = 255;
    }

    if( i < n )
    {
        const PaletteEntry* p0 = palette + (indices[0] >> 4);
        data[dblue_idx] = p0->b;
        data[1] = p0->g;
        data[dblue_idx^2] = p0->r;
        if( dcn == 4 )
            data[3] = 255;
    }
}


static void fillGrayRow4( uchar* data, const uchar* indices, int n, const uchar* palette )
{
    int i = 0;
    for( ; i <= n - 2; i += 2 )
    {
        int idx = *indices++;
        data[i] = palette[idx >> 4];
        data[i+1] = palette[idx & 15];
    }
    if( i < n )
        data[i] = palette[indices[0] >> 4];
}


static void fillColorRow1( uchar* data, const uchar* indices, int n,
                           const PaletteEntry* palette, int dcn, int dblue_idx )
{
    int i = 0, mask = 0, idx = 0;
    for( i = 0; i < n; i++, data += dcn, mask >>= 1 )
    {
        const PaletteEntry* p;
        if( mask == 0 )
        {
            idx = *indices++;
            mask = 128;
        }
        p = palette + ((idx & mask) != 0);
        data[dblue_idx] = p->b;
        data[1] = p->g;
        data[dblue_idx^2] = p->r;
        data[3] = 255;
    }
}


static void fillGrayRow1( uchar* data, const uchar* indices, int n, const uchar* palette )
{
    int i = 0, mask = 0, idx = 0;
    for( i = 0; i < n; i++, mask >>= 1 )
    {
        if( mask == 0 )
        {
            idx = *indices++;
            mask = 128;
        }
        data[i] = palette[(idx & mask) != 0];
    }
}

static int readBmpData(struct GrfmtReader* _reader, uchar* data, int step, int dcn )
{
    GrfmtBmpReader* reader = (GrfmtBmpReader*)_reader;
    uchar  gray_palette[256];
    int result = 0;
    int color = dcn > 1;
    int y, src_step, width, height, bpp;
    const uchar* p;
    const PaletteEntry* palette = 0;

    if( !reader || !reader->m_data || reader->m_offset <= 0 )
        return -1;

    width = reader->m_width;
    height = reader->m_height;
    bpp = reader->m_bpp;
    palette = &reader->m_palette[0];

    src_step = ((width*(bpp != 15 ? bpp : 16) + 7)/8 + 3) & -4;
    p = reader->m_data + reader->m_offset;

    if( reader->m_offset + src_step*height > reader->m_datasize )
        return -1;

    if( reader->m_origin > 0 )
    {
        data += (height - 1)*step;
        step = -step;
    }

    if( color == 0 && bpp <= 8 )
    {
        cvtRGBToGray(&palette[0].b, &gray_palette[0], (1 << bpp), 4, 0);
    }

    switch( bpp )
    {
    /************************* 1 BPP ************************/
    case 1:
        for( y = 0; y < height; y++, data += step, p += src_step )
        {
            if( color )
                fillColorRow1( data, p, width, palette, dcn, 2 );
            else
                fillGrayRow1( data, p, width, gray_palette );
        }
        result = 0;
        break;

    /************************* 4 BPP ************************/
    case 4:
        for( y = 0; y < height; y++, data += step, p += src_step )
        {
            if( color )
                fillColorRow4( data, p, width, palette, dcn, 2 );
            else
                fillGrayRow4( data, p, width, gray_palette );
        }
        result = 0;
        break;

    /************************* 8 BPP ************************/
    case 8:
        for( y = 0; y < height; y++, data += step, p += src_step )
        {
            if( color )
                fillColorRow8( data, p, width, palette, dcn, 2 );
            else
                fillGrayRow8( data, p, width, gray_palette );
        }
        result = 0;
        break;
    /************************* 24 BPP ************************/
    case 24:
        for( y = 0; y < height; y++, data += step, p += src_step )
        {
            if( color )
                cvtRGBToRGB( p, data, width, 3, 0, dcn, 2 );
            else
                cvtRGBToGray( p, data, width, 3, 0 );
        }
        result = 0;
        break;
    /************************* 32 BPP ************************/
    case 32:
        for( y = 0; y < height; y++, data += step, p += src_step )
        {
            if( color )
                cvtRGBToRGB( p, data, width, 4, 0, dcn, 2 );
            else
                cvtRGBToGray( p, data, width, 4, 0 );
        }
        result = 0;
        break;
    default:
        assert(0);
    }

    return result;
}


static struct GrfmtReader* readBmpHeader(const uchar* data, int datasize, int* _width, int* _height, int* _iscolor)
{
    GrfmtBmpReader* reader = 0;
    int result = 0;
    int iscolor = 0;
    int width = 0, height = 0, bpp = 0;
    BmpCompression rle_code = BMP_RGB;
    PaletteEntry palette[256];
    int offset, size;
    int j, clrused = 0;
    const uchar* p = data + 10;

    if( !data || datasize < 56 || !_width || !_height || !_iscolor || data[0] != 'B' || data[1] != 'M' )
        return 0;

    offset = GET_DWORD(p);
    size = GET_DWORD(p);

    memset(&palette[0], 0, sizeof(palette));

    if( size >= 36 )
    {
        if( (int)((p - data) + size) >= datasize )
            return 0;

        width = GET_DWORD(p);
        height = GET_DWORD(p);
        bpp = GET_DWORD(p) >> 16;
        rle_code = (BmpCompression)GET_DWORD(p);
        p += 12;
        clrused = GET_DWORD(p);
        p += size - 36;

        if( width > 0 && height != 0 &&
           (((bpp == 1 || bpp == 4 || bpp == 8 ||
              bpp == 24 || bpp == 32 ) && rle_code == BMP_RGB) ||
            (bpp == 16 && (rle_code == BMP_RGB || rle_code == BMP_BITFIELDS)) ||
            (bpp == 4 && rle_code == BMP_RLE4) ||
            (bpp == 8 && rle_code == BMP_RLE8)))
        {
            iscolor = 1;
            result = 1;

            if( bpp <= 8 )
            {
                clrused = clrused == 0 ? (1 << bpp) : clrused;
                memcpy(&palette[0], p, clrused*4);
                p += clrused*4;
                iscolor = isColorPalette( palette, clrused );
            }
            else if( bpp == 16 && rle_code == BMP_BITFIELDS )
            {
                int redmask = GET_DWORD(p);
                int greenmask = GET_DWORD(p);
                int bluemask = GET_DWORD(p);

                if( bluemask == 0x1f && greenmask == 0x3e0 && redmask == 0x7c00 )
                    bpp = 15;
                else if( bluemask == 0x1f && greenmask == 0x7e0 && redmask == 0xf800 )
                    ;
                else
                    result = 0;
            }
            else if( bpp == 16 && rle_code == BMP_RGB )
                bpp = 15;
        }
    }
    else if( size == 12 )
    {
        width  = GET_DWORD(p);
        height = GET_DWORD(p);
        bpp    = GET_DWORD(p) >> 16;
        rle_code = BMP_RGB;

        if( width > 0 && height != 0 &&
           (bpp == 1 || bpp == 4 || bpp == 8 || bpp == 24 || bpp == 32 ))
        {
            if( bpp <= 8 )
            {
                clrused = 1 << bpp;
                for( j = 0; j < clrused; j++, p += 3 )
                {
                    palette[j].b = p[0];
                    palette[j].g = p[1];
                    palette[j].r = p[2];
                    palette[j].a = 255;
                }
                iscolor = isColorPalette( palette, clrused );
            }
            result = 1;
        }
    }

    if( result == 0 || (bpp == 15 || bpp == 16) || (rle_code != BMP_RGB && rle_code != BMP_BITFIELDS) )
        return 0;

    reader = (GrfmtBmpReader*)malloc(sizeof(*reader));
    if( !reader )
        return 0;
    memset(reader, 0, sizeof(*reader));
    reader->base.release = releaseBmpReader;
    reader->base.read = readBmpData;

    reader->m_bpp = bpp;
    reader->m_width = width;
    reader->m_height = height;
    reader->m_origin = height > 0 ? 1 : 0;
    reader->m_offset = offset;
    reader->m_rle_code = rle_code;

    reader->m_data = data;
    reader->m_datasize = datasize;
    reader->m_offset = offset;

    if( _width )
        *_width = width;
    if( _height )
        *_height = height;
    if( _iscolor )
        *_iscolor = iscolor;

    memcpy(&reader->m_palette[0], &palette[0], sizeof(palette));
    return (GrfmtReader*)reader;
}

#define PUT_DWORD(p, val) \
    (((p)[0] = (uchar)(val)), \
    ((p)[1] = (uchar)((val) >> 8)), \
    ((p)[2] = (uchar)((val) >> 16)), \
    ((p)[3] = (uchar)((val) >> 24)), \
     (p) += 4)

//////////////////////////////////////////////////////////////////////////////////////////
static int writeBMP( const char* filename, const uchar* img, int step, int width, int height, int channels0 )
{
    int channels = channels0 == 4 ? 3 : channels0;
    int width3 = width*channels;
    int i, y, fileStep = (width3 + 3) & -4;

    int  bitmapHeaderSize = 40;
    int  paletteSize = channels > 1 ? 0 : 1024;
    int  headerSize = 14 /* fileheader */ + bitmapHeaderSize + paletteSize;
    int  fileSize = fileStep*height + headerSize;
    uchar* buf = 0, *p = 0;

    FILE* f = fopen(filename, "wb");
    if(!f)
        return -1;

    buf = p = (uchar*)malloc(fileSize);
    p[0] = 'B';
    p[1] = 'M';
    p += 2;
    PUT_DWORD(p, fileSize);
    PUT_DWORD(p, 0);
    PUT_DWORD(p, headerSize);
    PUT_DWORD(p, bitmapHeaderSize);
    PUT_DWORD(p, width);
    PUT_DWORD(p, height);
    PUT_DWORD(p, 1 | (channels << 19));
    PUT_DWORD(p, BMP_RGB);
    memset(p, 0, 20);
    p += 20;

    if( channels == 1 )
    {
        for( i = 0; i < 256; i++, p += 4 )
        {
            p[0] = p[1] = p[2] = (uchar)i;
            p[3] = 255;
        }
    }

    for( y = height - 1; y >= 0; y--, p += fileStep )
    {
        if( channels0 == 1 )
            memcpy(p, img + step*y, width);
        else
        {
            const uchar* imgrow = img + step*y;
            uchar* dst = p;
            for( i = 0; i < width; i++, imgrow += channels0, dst += 3 )
            {
                dst[0] = imgrow[2];
                dst[1] = imgrow[1];
                dst[2] = imgrow[0];
            }
        }
        if( fileStep > width3 )
            memset(p + width3, 0, fileStep - width3);
    }

    fwrite(buf, 1, fileSize, f);
    fclose(f);
    free(buf);
    return 0;
}

CT_Image ct_read_bmp(const unsigned char* data, int datasize, int dcn)
{
    int w = 0, h = 0, iscolor = 0;
    GrfmtReader* bmpreader = readBmpHeader(data, (int)datasize, &w, &h, &iscolor);
    int ok = 0;
    CT_Image image = 0;

    if( dcn <= 0 )
        dcn = iscolor ? 3 : 1;

    if( bmpreader )
    {
        vx_uint32 width  = w;
        vx_uint32 height = h;
        vx_df_image format = dcn == 1 ? VX_DF_IMAGE_U8 : dcn == 3 ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_RGBX;

        image = ct_allocate_image(width, height, format);

        if( image )
            ok = bmpreader->read(bmpreader, image->data.y, (int)ct_stride_bytes(image), dcn) >= 0;
        bmpreader->release(bmpreader);
    }

    if( ok )
        return image;

    //ct_release_image(&image);
    return 0;
}


int ct_write_bmp(const char* filename, CT_Image image)
{
    if( image )
    {
        int channels = ct_channels(image->format);
        return writeBMP(filename, image->data.y, (int)ct_stride_bytes(image),
                        (int)image->width, (int)image->height, channels);
    }
    return -1;
}
