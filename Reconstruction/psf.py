#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class PSF(object):

    ##
    # Compute rotated covariance matrix which expresses the PSF of the slice in
    # the coordinates of the HR volume
    # \date       2017-11-01 16:16:20+0000
    #
    # \param      self            The object
    # \param      slice           Slice object which is aimed to be simulated
    #                             according to the slice acquisition model
    # \param      reconstruction  Stack object containing the HR volume
    #
    # \return     Covariance matrix U*Sigma_diag*U' where U represents the
    #             orthogonal trafo between slice and reconstruction
    #
    def get_covariance_matrix_in_reconstruction_space(self,slice_sitk,reconstruction_sitk):

        cov = self.get_covariance_matrix_in_reconstruction_space_sitk(reconstruction_sitk.sitk.GetDirection(),
                                                                      slice_sitk.sitk.GetDirection(),
                                                                      slice_sitk.sitk.GetSpacing())

        return cov

    ##
    # Gets the axis-aligned covariance matrix describing the PSF in
    # reconstruction space coordinates.
    #
    # \param      reconstruction_direction_sitk  Image header (sitk) direction
    #                                            of reconstruction space
    # \param      slice_direction_sitk           Image header (sitk) direction
    #                                            of slice space
    # \param      slice_spacing                  Spacing of slice space
    #
    # \return     Axis-aligned covariance matrix describing the PSF.
    #
    def get_covariance_matrix_in_reconstruction_space_sitk(self,
                                                           reconstruction_direction_sitk,
                                                           slice_direction_sitk,
                                                           slice_spacing):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the reconstruction space
        U = self._get_relative_rotation_matrix(slice_direction_sitk, reconstruction_direction_sitk)

        # Get axis-aligned PSF
        cov = self.get_gaussian_psf_covariance_matrix_from_spacing(slice_spacing)

        # Return Gaussian blurring variance covariance matrix of slice in
        # reconstruction space coordinates
        return U.dot(cov).dot(U.transpose())

    ##
    # Gets the predefined covariance matrix in reconstruction space.
    #
    # \param      reconstruction_direction_sitk  Image header (sitk) direction
    #                                            of reconstruction space
    # \param      slice_direction_sitk           Image header (sitk) direction
    #                                            of slice space
    # \param      cov                            Axis-aligned covariance matrix
    #                                            describing the PSF
    #
    # \return     The predefined covariance matrix in reconstruction space
    #             coordinates.
    #
    def get_predefined_covariance_matrix_in_reconstruction_space(self,
                                                                 reconstruction_direction_sitk,
                                                                 slice_direction_sitk,
                                                                 cov):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the reconstruction space
        U = self._get_relative_rotation_matrix(slice_direction_sitk, reconstruction_direction_sitk)

        # Return Gaussian blurring variance covariance matrix of slice in
        # reconstruction space coordinates
        return U.dot(cov).dot(U.transpose())

    @staticmethod
    def get_gaussian_psf_covariance_matrix_from_spacing(spacing):

        # Compute Gaussian to approximate in-plane PSF:
        sigma_x2 = (1.2*spacing[0])**2/(8*np.log(2))
        sigma_y2 = (1.2*spacing[1])**2/(8*np.log(2))
        sigma_z2 = spacing[2]**2/(8*np.log(2))

        return np.diag([sigma_x2, sigma_y2, sigma_z2])

    # Gets the relative rotation matrix to express slice-axis aligned
    # covariance matrix in coordinates of HR volume
   
    @staticmethod
    def _get_relative_rotation_matrix(slice_direction_sitk,reconstruction_direction_sitk):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the HR volume
        dim = np.sqrt(len(slice_direction_sitk)).astype('int')
        direction_matrix_reconstruction = np.array(reconstruction_direction_sitk).reshape(dim, dim)
        direction_matrix_slice = np.array(slice_direction_sitk).reshape(dim, dim)

        U = direction_matrix_reconstruction.transpose().dot(direction_matrix_slice)

        return U




